import requests

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

import pybreaker

# Circuit Breaker
breaker = pybreaker.CircuitBreaker(
    fail_max=5,
    reset_timeout=30,
)


@breaker
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1),
    retry=retry_if_exception_type(requests.RequestException),
)
def search_github(query: str):

    response = requests.get(
        "https://api.github.com/search/repositories",
        params={"q": query},
        timeout=10,
    )

    response.raise_for_status()

    return response.json()