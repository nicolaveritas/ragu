import os
from functools import partial
from operator import add
from pathlib import Path
from typing import Annotated, Any
import instructor
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, convert_to_openai_messages
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from langgraph.prebuilt import ToolNode

from ragu.prompt_loader import render_prompt

from langfuse import observe, propagate_attributes, get_client
from langfuse.langchain import CallbackHandler


langfuse_handler = CallbackHandler()

PROMPTS_DIR = Path(__file__).parent / "prompts"
CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
RICETTARIO_MCP_URL = os.getenv("RICETTARIO_MCP_URL", "http://localhost:8001/mcp")

mcp_client = MultiServerMCPClient(
    {"ricettario": {"url": RICETTARIO_MCP_URL, "transport": "http"}}
)


class RAGUsedContext(BaseModel):
    id: int = Field(description="Id of a recipe used in the answer, as returned by the search tool")
    name: str = Field(description="Name of the recipe, as returned by the search tool")


class FinalResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question, in natural language")
    references: list[RAGUsedContext] = Field(
        description=(
            "Every recipe that contributed to the answer, and only those. "
            "Do not include retrieved recipes you did not use."
        )
    )


class State(BaseModel):
    messages: Annotated[list[Any], add] = []
    question_relevant: bool = True
    iteration: int = 0
    answer: str = ""
    final_answer: bool = False
    references: list[RAGUsedContext] = []


llm = ChatOpenAI(
    model="gpt-5.4-mini",
    reasoning_effort="low",
    use_responses_api=True,
)


# agent_llm arrives from build_graph via partial: the tools it is bound to come from
# ricettario over MCP, so they don't exist yet at import time.
def agent_node(state: State, agent_llm) -> dict:
    prompt = render_prompt(PROMPTS_DIR, "agent_prompt")
    response = agent_llm.invoke([SystemMessage(content=prompt), *state.messages])

    final_answer = False
    answer = ""
    references = []
    for tc in response.tool_calls:
        if tc["name"] == "FinalResponse":
            final_answer = True
            answer = tc["args"]["answer"]
            references.extend(tc["args"]["references"])

    message = AIMessage(content=answer) if final_answer else response
    return {
        "messages": [message],
        "iteration": state.iteration + 1,
        "final_answer": final_answer,
        "answer": answer,
        "references": references,
    }


def tool_router(state: State) -> str:
    if state.final_answer:
        return "end"
    elif state.iteration > 3:  # allow recipes->reviews->answer (+1 stumble) before force-ending
        return "end"
    elif len(state.messages[-1].tool_calls) > 0:
        return "tools"
    else:
        return "end"


class IntentRouterResponse(BaseModel):
    question_relevant: bool
    answer: str = Field(
        description=(
            "A brief, friendly reply when question_relevant is false. "
            "Leave empty when question_relevant is true."
        )
    )


intent_client = instructor.from_provider(
    "openai/gpt-5.4-nano",
    mode=instructor.Mode.RESPONSES_TOOLS,
)


def intent_router_node(state: State) -> dict:
    response = intent_client.create(
        messages=[
            {"role": "system", "content": render_prompt(PROMPTS_DIR, "intent_router")},
            *convert_to_openai_messages([state.messages[-1]]),
        ],
        reasoning={"effort": "none"},
        response_model=IntentRouterResponse,
    )
    return {"question_relevant": response.question_relevant, "answer": response.answer}


def intent_router_conditional_edges(state: State) -> str:
    if state.question_relevant:
        return "agent_node"
    else:
        return "end"


async def build_graph():
    """Wire the graph around the tools ricettario advertises, asked for every run.

    The tool list costs one ~10ms round-trip against several seconds of agent work,
    so it isn't cached: editing a tool docstring in ricettario takes effect on the
    next question instead of needing a restart.
    """
    tools = await mcp_client.get_tools()
    agent_llm = llm.bind_tools([*tools, FinalResponse], tool_choice="required")
    return (
        StateGraph(State)
        .add_node("tool_node", ToolNode(tools))
        .add_node("intent_router_node", intent_router_node)
        .add_node("agent_node", partial(agent_node, agent_llm=agent_llm))
        .add_edge(START, "intent_router_node")
        .add_conditional_edges(
            "intent_router_node",
            intent_router_conditional_edges,
            {
                "agent_node": "agent_node",
                "end": END
            }
        )
        .add_conditional_edges(
            "agent_node",
            tool_router,
            {
                "tools": "tool_node",
                "end": END
            }
        )
        .add_edge("tool_node", "agent_node")
    )


TOOL_STATUS = {
    "search_recipes": "Searching recipes…",
    "search_reviews": "Reading what people cooked…",
}


def _status(update: dict) -> str:
    """Turn one `updates` chunk — {node_name: state_delta} — into a line for the UI.

    Read the tool calls the agent just produced, not the tools already run: the call
    names say what is about to happen, so the status lands before the wait, not after.
    """
    for node, delta in update.items():
        if node == "intent_router_node":
            return "Reading your question…"
        if node == "agent_node" and not delta["final_answer"]:
            names = [tc["name"] for tc in delta["messages"][-1].tool_calls]
            return " ".join(TOOL_STATUS.get(n, f"Running {n}…") for n in names) or "Thinking…"
    return ""


# Async because the MCP tools are async-only: a sync graph.invoke() can't call them.
@observe(name="run_agent")
async def stream_agent(question: str, thread_id: str):
    """Run the graph and yield events as they happen, instead of one dict at the end.

    Same run as before, only consumed step by step: astream hands back a chunk per
    node, so the seconds spent in retrieval become progress lines instead of a spinner.
    Two stream modes at once — `updates` (what a node just changed, for status text)
    and `values` (the whole state, whose last chunk is the final state).

    Yields {"type": "status", "text": ...} zero or more times, then exactly one
    {"type": "final", ...} carrying the same payload run_agent used to return.
    """
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "iteration": 0,
    }

    result = State()
    builder = await build_graph()
    with propagate_attributes(session_id=thread_id):
        async with AsyncPostgresSaver.from_conn_string(CONNECTION_STRING) as checkpointer:
            graph = builder.compile(checkpointer=checkpointer)
            async for chunk in graph.astream(
                initial_state,
                config={
                    "callbacks": [langfuse_handler],
                    "configurable": {"thread_id": thread_id}
                },
                stream_mode=["updates", "values"],
                version="v2",
            ):
                if chunk["type"] == "values":
                    result = chunk["data"]  # last one wins: the state at END
                elif text := _status(chunk["data"]):
                    yield {"type": "status", "text": text}

    # A `values` chunk is the State model, not the dict ainvoke returned: dump it so
    # references land as plain dicts whatever pydantic coerced them into.
    final = result.model_dump(include={"answer", "references", "question_relevant", "iteration"})

    # Relevant question but the loop hit the iteration cap without a FinalResponse:
    # never hand the UI a blank bubble. (Off-topic path already carries the router's reply.)
    answer = final["answer"] or "Sorry, I couldn't put that together — could you rephrase or narrow it down?"

    yield {
        "type": "final",
        "question": question,
        **final,
        "answer": answer,
        "trace_id": get_client().get_current_trace_id(),
    }


async def run_agent(question: str, thread_id: str) -> dict:
    """Blocking shortcut for callers with nothing to show mid-run (evals, notebooks)."""
    async for event in stream_agent(question, thread_id):
        if event["type"] == "final":
            return event