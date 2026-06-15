from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.chains import GradeResult, grade_chain, rag_chain, rewrite_chain
from app.config import settings
from app.vectorstore import build_vectorstore, get_retriever


# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # full conversation history
    retrieved_docs: list[str]                  # text of retrieved chunks
    current_question: str                      # may be rewritten on retries
    quality_score: int                         # last evaluator score
    retry_count: int                           # how many retries so far


# Build vector store once at module load
_vectorstore = build_vectorstore()
_retriever = get_retriever(_vectorstore)


# ── Nodes ─────────────────────────────────────────────────────────────────────
def retrieve_node(state: AgentState) -> dict:
    docs = _retriever.invoke(state["current_question"])
    return {"retrieved_docs": [doc.page_content for doc in docs]}


def generate_node(state: AgentState) -> dict:
    context = "\n\n---\n\n".join(state["retrieved_docs"])
    history = state["messages"][:-1]  # everything except the current human turn
    answer = rag_chain.invoke({
        "question": state["current_question"],
        "context": context,
        "history": history,
    })
    return {"messages": [AIMessage(content=answer)]}


def evaluate_node(state: AgentState) -> dict:
    last_ai = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
    result: GradeResult = grade_chain.invoke({
        "question": state["current_question"],
        "answer": last_ai.content,
    })
    print(f"[evaluate] score={result.score}/10 — {result.reasoning}")
    return {"quality_score": result.score}


def rewrite_node(state: AgentState) -> dict:
    rewritten = rewrite_chain.invoke({
        "question": state["current_question"],
        "history": state["messages"],
    })
    print(f"[rewrite] '{state['current_question']}' -> '{rewritten}'")
    return {
        "current_question": rewritten,
        "retry_count": state["retry_count"] + 1,
    }


# ── Routing ───────────────────────────────────────────────────────────────────
def route_after_evaluate(state: AgentState) -> Literal["rewrite", "__end__"]:
    if state["quality_score"] >= settings.grade_threshold:
        print(f"[route] score {state['quality_score']} >= {settings.grade_threshold} -> END")
        return "__end__"
    if state["retry_count"] >= settings.max_retries:
        print(f"[route] max retries reached -> END anyway")
        return "__end__"
    print(f"[route] score {state['quality_score']} too low, retrying...")
    return "rewrite"


# ── Build graph ───────────────────────────────────────────────────────────────
def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("rewrite",  rewrite_node)

    builder.add_edge(START,      "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "evaluate")
    builder.add_edge("rewrite",  "retrieve")

    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"__end__": END, "rewrite": "rewrite"},
    )

    return builder.compile(checkpointer=MemorySaver())


graph = build_graph()


# ── Public API ────────────────────────────────────────────────────────────────
def chat(question: str, session_id: str) -> str:
    """Send a message and get a response. Same session_id = same conversation."""
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": settings.recursion_limit,
    }
    final_state = graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "current_question": question,
            "quality_score": 0,
            "retry_count": 0,
            "retrieved_docs": [],
        },
        config=config,
    )
    last_ai = next(m for m in reversed(final_state["messages"]) if isinstance(m, AIMessage))
    return last_ai.content
