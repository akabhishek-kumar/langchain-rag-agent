from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from app.config import settings

# Single LLM instance — model from config, never hardcoded
llm = ChatGroq(
    model=settings.groq_model,
    temperature=0,
    api_key=settings.groq_api_key,
)

# ── 1. RAG chain: question + context + history -> answer string ───────────────
rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a knowledgeable assistant. Answer the question using ONLY the "
        "provided context. If the context does not contain enough information, "
        "say so clearly.\n\nContext:\n{context}",
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])
rag_chain = rag_prompt | llm | StrOutputParser()


# ── 2. Grade chain: question + answer -> GradeResult (score + reasoning) ─────
class GradeResult(BaseModel):
    score: int = Field(description="Relevance and accuracy score 1-10", ge=1, le=10)
    reasoning: str = Field(description="One sentence explaining the score")


grade_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict quality evaluator. Rate the answer 1-10 for relevance "
        "and accuracy. 7+ means the answer fully addresses the question.",
    ),
    ("human", "Question: {question}\n\nAnswer: {answer}\n\nProvide score and reasoning."),
])
grade_chain = grade_prompt | llm.with_structured_output(GradeResult)


# ── 3. Rewrite chain: question + history -> better standalone question ────────
rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Rewrite the user's question to be more specific and self-contained, "
        "resolving any pronouns using the conversation history. "
        "Output ONLY the rewritten question.",
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "Original question: {question}\n\nRewritten question:"),
])
rewrite_chain = rewrite_prompt | llm | StrOutputParser()
