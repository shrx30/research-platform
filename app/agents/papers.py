from app.Tools.papers_tools import search_papers
from app.agents.relevance import evaluate_relevance


RELEVANCE_THRESHOLD = 0.7


def run(query: str, user_query: str) -> str:

    papers = search_papers(query)

    if not papers:
        return "No papers found."

    contexts = []

    for paper in papers:

        evaluation_content = f"""
Title:
{paper["title"]}

Abstract:
{paper["summary"]}
"""

        try:
            evaluation = evaluate_relevance(
                user_query=user_query,
                content=evaluation_content,
            )

        except Exception as exc:
            print(
                f"[PAPERS] Relevance check failed for "
                f"'{paper['title']}': {exc}"
            )

            # Keep result if evaluator itself failed
            continue

        keep = evaluation.score >= RELEVANCE_THRESHOLD

        print(
            f"[PAPERS] {paper['title'][:60]} "
            f"→ {evaluation.score:.2f} "
            f"({'KEEP' if keep else 'DROP'})"
        )

        if not keep:
            continue

        contexts.append(
            f"""
Title:
{paper["title"]}

Authors:
{", ".join(paper["authors"])}

Published:
{paper["published"]}

URL:
{paper["url"]}

Abstract:
{paper["summary"]}
"""
        )

    if not contexts:
        return "No relevant academic papers found."

    return "\n\n---------------------------\n\n".join(contexts)