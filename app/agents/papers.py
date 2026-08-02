from app.Tools.papers_tools import search_papers


def run(task: str) -> str:
    """Search arXiv and return paper results as text."""

    papers = search_papers(task, limit=5)

    if not papers:
        return "No research papers found."

    contexts = []

    for paper in papers:
        authors = paper.get("authors", [])

        if isinstance(authors, list):
            authors = ", ".join(authors)

        contexts.append(
            f"""
Title: {paper.get("title", "Unknown")}

Authors: {authors}

Published: {paper.get("published", "Unknown")}

Summary:
{paper.get("summary", "")}

URL: {paper.get("url", "")}
"""
        )

    return "\n\n-------------------------\n\n".join(contexts)