from app.Tools.web_tools import search_web
def run(task: str) -> str:
    """Execute a web search task."""

    try:
        results = search_web(task)

    except Exception as exc:
        return f"Web search failed: {exc}"

    if not results:
        return "No web results found."

    contexts = []

    for result in results:

        contexts.append(
            f"""
Title: {result.get("title")}

URL: {result.get("url")}

Content:
{result.get("content")}
"""
        )

    return (
        "\n\n-------------------------\n\n"
        .join(contexts)
    )