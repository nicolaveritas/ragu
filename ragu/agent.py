import json
from operator import add
from pathlib import Path
from typing import Annotated, Any
import instructor
from langchain_core.messages import AIMessage, HumanMessage, convert_to_openai_messages
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from langgraph.prebuilt import ToolNode

from ragu.retrieval import format_blocks, rerank, retrieve_data
from ragu.prompt_loader import render_prompt

from langfuse import observe
from langfuse.langchain import CallbackHandler
from langfuse.openai import openai

langfuse_handler = CallbackHandler()

PROMPTS_DIR = Path(__file__).parent / "prompts"

@tool
def search_recipes(query: str, top_k: int = 5) -> str:
    """Search the recipe database and return the most relevant recipes.

    Args:
        query: Natural-language search query. Keep any numeric constraints in
            the query text (e.g. "under 300 calories", "at least 20g protein",
            "ready in 30 minutes"): they are extracted automatically and
            applied as hard filters on the search.
        top_k: Number of recipes to return. Works best with 5 or more.

    Returns:
        One text block per recipe with name, id, rating, nutrition facts,
        total time, ingredients and steps. Starts with a note if no recipe
        satisfied the numeric constraints (closest matches are shown instead).
    """
    retrieved = retrieve_data(query, k=20)
    recipes = rerank(query, retrieved["recipes"], top_n=top_k)
    if not recipes:
        return "No recipes found for this query. Try rephrasing or broadening it."
    text = "\n\n".join(format_blocks(recipes))
    if retrieved["filter_relaxed"]:
        text = (
            "Note: no recipes matched the numeric constraints; "
            "showing the closest matches instead.\n\n" + text
        )
    return text


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


# OpenAI tool schemas: what bind_tools would have built from the docstrings/models
AGENT_TOOLS = [convert_to_openai_tool(search_recipes), convert_to_openai_tool(FinalResponse)]


def agent_node(state: State) -> dict:
    prompt = render_prompt(PROMPTS_DIR, "agent_prompt")
    response = openai.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[
            {"role": "system", "content": prompt},
            *convert_to_openai_messages(state.messages),
        ],
        tools=AGENT_TOOLS,
        tool_choice="required",
        reasoning_effort="none",
        name="agent_llm",
    )
    msg = response.choices[0].message
    # back to LangChain-land: ToolNode wants an AIMessage with parsed tool_calls
    tool_calls = [
        {
            "name": tc.function.name,
            "args": json.loads(tc.function.arguments),
            "id": tc.id,
            "type": "tool_call",
        }
        for tc in (msg.tool_calls or [])
    ]

    final_answer = False
    answer = ""
    references = []
    for tc in tool_calls:
        if tc["name"] == "FinalResponse":
            final_answer = True
            answer = tc["args"]["answer"]
            references.extend(tc["args"]["references"])

    return {
        "messages": [AIMessage(content=msg.content or "", tool_calls=tool_calls)],
        "iteration": state.iteration + 1,
        "final_answer": final_answer,
        "answer": answer,
        "references": references,
    }


def tool_router(state: State) -> str:
    if state.final_answer:
        return "end"
    elif state.iteration > 2:
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
            *convert_to_openai_messages(state.messages),
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


graph = (
    StateGraph(State)
    .add_node("tool_node", ToolNode([search_recipes]))
    .add_node("intent_router_node", intent_router_node)
    .add_node("agent_node", agent_node)
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
    .compile()
)


@observe(name="run_agent")
def run_agent(question: str) -> dict:
    initial_state = {
        "messages": [HumanMessage(content=question)],
    }

    result = graph.invoke(initial_state, config={"callbacks": [langfuse_handler]})

    # Relevant question but the loop hit the iteration cap without a FinalResponse:
    # never hand the UI a blank bubble. (Off-topic path already carries the router's reply.)
    answer = result.get("answer") or "Sorry, I couldn't put that together — could you rephrase or narrow it down?"

    return {
      "question": question,
      "answer": answer,
      "references": result.get("references", []),
      "question_relevant": result.get("question_relevant", True),
      "iteration": result.get("iteration", 0),
  }