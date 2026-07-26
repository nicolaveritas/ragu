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


# Async because the MCP tools are async-only: a sync graph.invoke() can't call them.
@observe(name="run_agent")
async def run_agent(question: str, thread_id: str) -> dict:
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "iteration": 0,
    }

    builder = await build_graph()
    with propagate_attributes(session_id=thread_id):
        async with AsyncPostgresSaver.from_conn_string(CONNECTION_STRING) as checkpointer:
            graph = builder.compile(checkpointer=checkpointer)
            result = await graph.ainvoke(
                initial_state,
                config={
                    "callbacks": [langfuse_handler],
                    "configurable": {"thread_id": thread_id}
                }
            )

    # Relevant question but the loop hit the iteration cap without a FinalResponse:
    # never hand the UI a blank bubble. (Off-topic path already carries the router's reply.)
    answer = result.get("answer") or "Sorry, I couldn't put that together — could you rephrase or narrow it down?"

    return {
      "question": question,
      "answer": answer,
      "references": result.get("references", []),
      "question_relevant": result.get("question_relevant", True),
      "iteration": result.get("iteration", 0),
      "trace_id": get_client().get_current_trace_id(),
  }