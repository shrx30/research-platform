from typing import TypedDict
from app.schemas.research import ResearchResult


class ResearchState(TypedDict):
    # User input
    query: str

    # Planner output
    plan: list[dict]

    # Agent outputs
    web_context: str
    github_context: str
    paper_context: str
    memory_context: str

    # Merge output
    merged_context: str

    # Final outputs
    research_result: ResearchResult
    report: str