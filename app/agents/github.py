from app.Tools.github_tools import search_github
def run(task: str) -> str:
    """
    Execute a GitHub search task and convert
    repositories into context for the LLM.
    """

    repos = search_github(task)

    if not repos:
        return "No GitHub repositories found."

    contexts = []

    for repo in repos:

        contexts.append(
            f"""
Repository: {repo["full_name"]}

Description:
{repo["description"]}

Stars:
{repo["stargazers_count"]}

Language:
{repo["language"]}

URL:
{repo["html_url"]}
"""
        )

    return (
        "\n\n-----------------------------\n\n"
        .join(contexts)
    )