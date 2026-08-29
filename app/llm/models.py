from langchain_nvidia_ai_endpoints import ChatNVIDIA

from app.config.settings import settings
from app.schemas.planner import ExecutionPlan
from app.schemas.research import ResearchResult


# =========================================================
# FAST GENERAL MODEL
# =========================================================

llm = ChatNVIDIA(
    model="deepseek-ai/deepseek-v4-flash",
    api_key=settings.NVIDIA_API_KEY,
    temperature=0,
    max_tokens=2000,
    timeout=60,
)


# =========================================================
# KIMI K3
# =========================================================

kimi_k3 = ChatNVIDIA(
    model="moonshotai/kimi-k3",
    api_key=settings.NVIDIA_API_KEY,
    temperature=0,
    max_tokens=3000,
    timeout=90,
)


# =========================================================
# PLANNER
# =========================================================

planner_llm = kimi_k3.with_structured_output(
    ExecutionPlan
)


# =========================================================
# RESEARCH SYNTHESIS
# =========================================================

research_llm = kimi_k3.with_structured_output(
    ResearchResult
)


# =========================================================
# RELEVANCE
# =========================================================

relevance_llm = llm
