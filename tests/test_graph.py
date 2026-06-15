from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage
from app.chains import GradeResult


def make_mocks(grade_score=8):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        MagicMock(page_content="Coded Agents use LangGraph."),
        MagicMock(page_content="MCP tools use @mcp.tool()."),
    ]
    mock_rag = MagicMock()
    mock_rag.invoke.return_value = "A Coded Agent uses LangGraph to orchestrate LLM calls."
    mock_grade = MagicMock()
    mock_grade.invoke.return_value = GradeResult(score=grade_score, reasoning="Test.")
    mock_rewrite = MagicMock()
    mock_rewrite.invoke.return_value = "More specific question?"
    return mock_retriever, mock_rag, mock_grade, mock_rewrite


def test_happy_path():
    mock_retriever, mock_rag, mock_grade, mock_rewrite = make_mocks(grade_score=8)
    with (
        patch("app.graph._retriever", mock_retriever),
        patch("app.graph.rag_chain", mock_rag),
        patch("app.graph.grade_chain", mock_grade),
        patch("app.graph.rewrite_chain", mock_rewrite),
    ):
        from app.graph import build_graph
        g = build_graph()
        state = g.invoke(
            {"messages": [HumanMessage(content="What is a Coded Agent?")],
             "current_question": "What is a Coded Agent?",
             "quality_score": 0, "retry_count": 0, "retrieved_docs": []},
            config={"configurable": {"thread_id": "t1"}},
        )
    assert any(isinstance(m, AIMessage) for m in state["messages"])
    mock_rewrite.invoke.assert_not_called()
    assert state["quality_score"] == 8


def test_retry_on_low_score():
    mock_retriever, mock_rag, _, mock_rewrite = make_mocks()
    call_n = {"n": 0}
    mock_grade = MagicMock()
    def grade_side(inp):
        call_n["n"] += 1
        return GradeResult(score=3 if call_n["n"] == 1 else 8, reasoning="Test.")
    mock_grade.invoke.side_effect = grade_side

    with (
        patch("app.graph._retriever", mock_retriever),
        patch("app.graph.rag_chain", mock_rag),
        patch("app.graph.grade_chain", mock_grade),
        patch("app.graph.rewrite_chain", mock_rewrite),
    ):
        from app.graph import build_graph
        g = build_graph()
        state = g.invoke(
            {"messages": [HumanMessage(content="What is RAG?")],
             "current_question": "What is RAG?",
             "quality_score": 0, "retry_count": 0, "retrieved_docs": []},
            config={"configurable": {"thread_id": "t2"}},
        )
    assert mock_rewrite.invoke.call_count == 1
    assert state["retry_count"] == 1
