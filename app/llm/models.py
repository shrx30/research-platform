from langchain_nvidia_ai_endpoints import ChatNVIDIA

from app.config.settings import settings
from app.schemas.planner import ExecutionPlan
from app.schemas.research import ResearchResult


# General reasoning model
llm = ChatNVIDIA(
    model="deepseek-ai/deepseek-v4-flash",
    api_key=settings.NVIDIA_API_KEY,
    temperature=0,
    timeout=180,
)


# Structured-output model
structured_llm = ChatNVIDIA(
    model="openai/gpt-oss-120b",
    api_key=settings.NVIDIA_API_KEY,
    temperature=0,
    timeout=180,
)


planner_llm = structured_llm.with_structured_output(
    ExecutionPlan
)

research_llm = structured_llm.with_structured_output(
    ResearchResult
)


# Cheap relevance classification
relevance_llm = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct",
    api_key=settings.NVIDIA_API_KEY,
    temperature=0,
    max_tokens=200,
    timeout=60,
)