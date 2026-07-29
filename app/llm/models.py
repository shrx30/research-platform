from langchain_nvidia_ai_endpoints import ChatNVIDIA

from app.config.settings import settings
from app.schemas.planner import ExecutionPlan
from app.schemas.research import ResearchResult


# =========================================================
# GENERAL REASONING MODEL
# =========================================================

llm = ChatNVIDIA(
    model="deepseek-ai/deepseek-v4-flash",
    api_key=settings.NVIDIA_API_KEY,
    temperature=0,
    max_tokens=3000,
    timeout=180,
)


# =========================================================
# STRUCTURED OUTPUT BASE MODEL
# =========================================================

structured_base = ChatNVIDIA(
    model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
    api_key=settings.NVIDIA_API_KEY,
    temperature=0,
    max_tokens=3000,
    timeout=180,
)


# =========================================================
# PLANNER
# =========================================================

planner_llm = structured_base.with_structured_output(
    ExecutionPlan
)


# =========================================================
# RESEARCH
# =========================================================

research_llm = structured_base.with_structured_output(
    ResearchResult
)


# Normal model exposed for JSON fallback.
research_base = structured_base


# =========================================================
# RELEVANCE CLASSIFIER
# =========================================================

relevance_llm = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct",
    api_key=settings.NVIDIA_API_KEY,
    temperature=0,
    max_tokens=200,
    timeout=60,
)