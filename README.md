# langchain-rag-agent-v2

A stateful **LangGraph RAG Agent** built with LangChain, Groq (free LLM), and FastEmbed embeddings. Implements the **Planner-Generator-Evaluator** pattern with automatic query rewriting and multi-turn conversation memory.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Framework | LangGraph 0.2+ |
| LLM | Groq — llama-3.1-8b-instant (free tier) |
| Embeddings | FastEmbed BAAI/bge-small-en-v1.5 (ONNX, no PyTorch) |
| Vector Store | Chroma (in-memory) |
| API | FastAPI + uvicorn |
| Config | pydantic-settings (.env) |

## Architecture — Planner-Generator-Evaluator Pattern

```
User Question
      │
      ▼
  [retrieve]  ── fetches top-k chunks from Chroma
      │
      ▼
  [generate]  ── RAG chain: context + history → LLM → answer
      │
      ▼
  [evaluate]  ── grades answer 1-10 for relevance & accuracy
      │
      ├── score ≥ 7 ──► END (return answer)
      │
      └── score < 7 ──► [rewrite] ── rewrites question ──► [retrieve]
                         (max 2 retries, then END)
```

This pattern prevents **self-evaluation bias** — the Generator and Evaluator are separate nodes so the model cannot grade its own output.

## Project Structure

```
app/
├── config.py       # pydantic-settings — GROQ_API_KEY, thresholds, recursion_limit
├── vectorstore.py  # FastEmbed embeddings + Chroma in-memory store
├── chains.py       # rag_chain, grade_chain, rewrite_chain (LCEL pipes)
├── graph.py        # LangGraph StateGraph — 4 nodes, AgentState, MemorySaver
└── api.py          # POST /chat/ endpoint
main.py             # FastAPI entry point
tests/
└── test_graph.py   # unit tests with mocked LLM calls
```

## Key Concepts Demonstrated

- **LangGraph StateGraph** — `TypedDict` state, `add_messages` reducer, `MemorySaver` with `thread_id`
- **LCEL (LangChain Expression Language)** — `|` pipe operator, every component is a `Runnable`
- **Harness Engineering** — Tool Registry, Model Management via `.env`, Context Management via `MessagesState`, Guardrails via `recursion_limit` + `max_retries`, Verification via Evaluator node
- **FastEmbed** — ONNX-based embeddings, no PyTorch dependency, no GPU required
- **Groq free tier** — no OpenAI billing needed, `llama-3.1-8b-instant` via `langchain-groq`

## Quick Start

```bash
git clone https://github.com/akabhishek-kumar/langchain-rag-agent-v2
cd langchain-rag-agent-v2
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env          # add your Groq API key
uvicorn main:app --reload
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

## API

```bash
POST /chat/
{
  "message": "What is Harness Engineering?",
  "session_id": "user-123"
}
```

## Running Tests

```bash
pytest tests/ -v
```

---

Part of my AI Engineering learning series → [GitHub](https://github.com/akabhishek-kumar)
