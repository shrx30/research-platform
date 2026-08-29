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
    max_tokens=2000,
    timeout=60,
)


# =========================================================
# PLANNER MODEL
# =========================================================
#
# Planner only needs to produce a small execution plan.
#
# Use the currently available Nemotron model instead of
# the retired Llama 3.1 8B endpoint.
#
# =========================================================

planner_base = ChatNVIDIA(
    model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
    api_key=settings.NVIDIA_API_KEY,
    temperature=0,
    max_tokens=700,
    timeout=60,
)


planner_llm = planner_base.with_structured_output(
    ExecutionPlan
)


# =========================================================
# STRUCTURED FALLBACK MODEL
# =========================================================
#
# Used when structured planner generation needs a fallback.
#
# =========================================================

structured_base = ChatNVIDIA(
    model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
    api_key=settings.NVIDIA_API_KEY,
    temperature=0,
    max_tokens=1000,
    timeout=60,
)


# =========================================================
# RESEARCH / SYNTHESIS MODEL
# =========================================================
#
# Research synthesis needs more context and reasoning
# capability than the planner.
#
# =========================================================

research_base = ChatNVIDIA(
    model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
    api_key=settings.NVIDIA_API_KEY,
    temperature=0,
    max_tokens=2000,
    timeout=60,
)


research_llm = research_base.with_structured_output(
    ResearchResult
)


# =========================================================
# RELEVANCE CLASSIFIER
# =========================================================
#
# Small classification task.
#
# For now use the general reasoning model rather than
# depending on another NVIDIA model that may be retired.
#
# =========================================================

relevance_llm = llm
