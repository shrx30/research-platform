from typing import TypedDict

from app.schemas.research import ResearchResult


class ResearchState(TypedDict, total=False):

    # User input
    query: str

    # Planner output
    plan: list

    # Agent outputs
    web_results: str
    github_results: str
    paper_results: str
    memory_results: str

    # Merged research context
    merged_context: str

    # Final structured result
    research_result: ResearchResult

    # Final report
    report: str