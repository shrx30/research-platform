from tavily import TavilyClient

from app.config.settings import settings

client = TavilyClient(api_key=settings.TAVILY_API_KEY)


def search_web(query: str, max_results: int = 5):
    """Search the web using Tavily."""

    try:
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
        )

        return response["results"]

    except Exception as e:
        raise RuntimeError(f"Tavily search failed: {e}")