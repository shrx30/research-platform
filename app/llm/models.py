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
# PLANNER BASE MODEL
# =========================================================
#
# Planning output is tiny.
# Do NOT give the planner a 180 second timeout.
#

planner_base = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct",
    api_key=settings.NVIDIA_API_KEY,
    temperature=0,
    max_tokens=700,
    timeout=30,
)


# Structured planner
planner_llm = planner_base.with_structured_output(
    ExecutionPlan
)


# =========================================================
# STRUCTURED FALLBACK MODEL
# =========================================================
#
# Used if structured planner generation fails.
#

structured_base = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct",
    api_key=settings.NVIDIA_API_KEY,
    temperature=0,
    max_tokens=1000,
    timeout=40,
)


# =========================================================
# RESEARCH / SYNTHESIS MODEL
# =========================================================
#
# Synthesis needs more capability than routing,
# so keep Nemotron here.
#

research_base = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct",
    api_key=settings.NVIDIA_API_KEY,
    temperature=0,
    max_tokens=1500,
    timeout=60,
)


research_llm = research_base.with_structured_output(
    ResearchResult
)


# =========================================================
# RELEVANCE CLASSIFIER
# =========================================================
#
# Classification is a small-model task.
#

relevance_llm = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct",
    api_key=settings.NVIDIA_API_KEY,
    temperature=0,
    max_tokens=200,
    timeout=30,
)