from fastapi import FastAPI
from app.api import router
from app.config import settings

app = FastAPI(
    title="LangGraph RAG Agent",
    description="Multi-turn RAG assistant — LangChain LCEL chains inside a LangGraph Planner-Generator-Evaluator graph.",
    version="0.1.0",
)

app.include_router(router)

@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok", "model": settings.groq_model}
