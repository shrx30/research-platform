import arxiv


def search_papers(
    query: str,
    limit: int = 5,
) -> list[dict]:

    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=limit,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []

    for result in client.results(search):

        papers.append(
            {
                "title": result.title,
                "authors": [
                    author.name
                    for author in result.authors
                ],
                "summary": result.summary,
                "published": result.published.strftime(
                    "%Y-%m-%d"
                ),
                "url": result.entry_id,
            }
        )

    return papers