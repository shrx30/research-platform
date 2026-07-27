import requests

from app.config.settings import settings


BASE_URL = "https://api.github.com/search/repositories"


def search_github(query: str, per_page: int = 5):
    """
    Search GitHub repositories.
    """

    headers = {
        "Accept": "application/vnd.github+json"
    }

    # Optional: use token to avoid rate limits
    if settings.GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {settings.GITHUB_TOKEN}"

    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": per_page
    }

    response = requests.get(
        BASE_URL,
        headers=headers,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return response.json()["items"]