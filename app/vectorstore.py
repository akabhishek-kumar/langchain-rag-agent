from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document

from app.config import settings

SAMPLE_DOCS = [
    Document(
        page_content=(
            "UiPath Coded Agents are code-first automation agents built using the UiPath SDK "
            "and Python or C#. Unlike low-code Studio workflows, Coded Agents use full "
            "programming constructs: loops, conditionals, async/await, and libraries like "
            "LangChain and LangGraph. A Coded Agent consists of a LangGraph state graph "
            "where each node performs a discrete action — calling an LLM, invoking a "
            "UiPath activity, or querying a database."
        ),
        metadata={"source": "uipath-coded-agents"},
    ),
    Document(
        page_content=(
            "The Model Context Protocol (MCP) is an open standard for exposing tools to LLMs. "
            "An MCP server registers tools using the @mcp.tool() decorator from FastMCP. "
            "Each tool has a name, description, and typed input schema. When a UiPath Coded "
            "Agent calls bind_tools(tools), the LLM receives tool definitions and can invoke "
            "them by returning a ToolCall. The agent executes the tool and feeds the result "
            "back as a ToolMessage."
        ),
        metadata={"source": "mcp-overview"},
    ),
    Document(
        page_content=(
            "Retrieval-Augmented Generation (RAG) grounds LLM responses in retrieved documents. "
            "The pipeline has three stages: ingestion (load documents, chunk them, embed each "
            "chunk, store in a vector DB), retrieval (embed the user question, find top-k "
            "similar chunks by cosine similarity), and generation (insert retrieved chunks as "
            "context into the prompt, then call the LLM). RAG reduces hallucinations because "
            "the LLM answers from provided text rather than parametric memory."
        ),
        metadata={"source": "rag-explainer"},
    ),
    Document(
        page_content=(
            "LangGraph builds stateful multi-step agents as directed graphs. A StateGraph "
            "defines nodes (Python functions that read and update shared state) and edges "
            "(connections between nodes, which can be conditional). MessagesState is a built-in "
            "state class with a messages list that uses an add_messages reducer — appending "
            "new messages rather than replacing the whole list. MemorySaver persists graph "
            "state across invocations using a thread_id, enabling multi-turn conversations."
        ),
        metadata={"source": "langgraph-guide"},
    ),
    Document(
        page_content=(
            "Harness Engineering is the third layer of AI engineering above prompt and context "
            "engineering. It designs the execution environment around AI agents: Tool Registry "
            "(what tools the agent can call), Model Management (which LLM and configuration), "
            "Context Management (state passing and window limits), Guardrails (recursion limits, "
            "retry counters, timeouts), and Verification Steps (post-execution checks independent "
            "of the LLM to confirm output correctness)."
        ),
        metadata={"source": "harness-engineering"},
    ),
    Document(
        page_content=(
            "The Planner-Generator-Evaluator pattern is a three-node LangGraph architecture. "
            "The Planner node breaks a task into sub-tasks or reformulates the question. "
            "The Generator node produces a response using a RAG chain or tool calls. "
            "The Evaluator node grades the output on relevance and accuracy, outputting a "
            "numeric score. A conditional edge routes back to the Planner if score is below "
            "threshold, or to END if acceptable. This prevents self-evaluation bias."
        ),
        metadata={"source": "planner-generator-evaluator"},
    ),
]


def build_vectorstore() -> Chroma:
    """Build Chroma vector store with FastEmbed (ONNX-based, no PyTorch needed)."""
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return Chroma.from_documents(
        documents=SAMPLE_DOCS,
        embedding=embeddings,
        collection_name="rag_agent_docs",
    )


def get_retriever(vectorstore: Chroma):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retriever_k},
    )
