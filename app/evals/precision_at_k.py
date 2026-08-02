from app.Tools.github_tools import search_github
from app.Tools.papers_tools import search_papers
from app.Tools.web_tools import search_web


TEST_CASES = [
    {
        "agent": "github",
        "query": "Vision Transformer implementation",
        "keywords": ["vision", "transformer"],
    },
    {
        "agent": "github",
        "query": "Retrieval Augmented Generation RAG",
        "keywords": ["retrieval", "rag"],
    },
    {
        "agent": "papers",
        "query": "Retrieval Augmented Generation",
        "keywords": ["retrieval", "generation"],
    },
    {
        "agent": "papers",
        "query": "multi-agent memory systems",
        "keywords": ["agent", "memory"],
    },
    {
        "agent": "web",
        "query": "LangGraph persistence",
        "keywords": ["langgraph", "persistence"],
    },
    {
        "agent": "web",
        "query": "AI agent memory architecture",
        "keywords": ["agent", "memory"],
    },
]


SEARCH_FUNCTIONS = {
    "github": search_github,
    "papers": search_papers,
    "web": search_web,
}


# =========================================================
# RESULT -> TEXT
# =========================================================


def result_to_text(result: dict) -> str:
    """
    Convert different retrieval result formats into
    one searchable text representation.
    """

    values = []

    for value in result.values():

        if value is None:
            continue

        if isinstance(value, list):
            values.extend(
                str(item)
                for item in value
            )

        else:
            values.append(
                str(value)
            )

    return " ".join(values).lower()


# =========================================================
# RELEVANCE
# =========================================================


def is_relevant(
    result: dict,
    keywords: list[str],
) -> bool:
    """
    Simple V1 relevance judge.

    A result is considered relevant when at least
    one expected keyword appears in the result.
    """

    text = result_to_text(
        result
    )

    return any(
        keyword.lower() in text
        for keyword in keywords
    )


# =========================================================
# PRECISION@K
# =========================================================


def precision_at_k(
    results: list,
    keywords: list[str],
    k: int = 5,
) -> float:

    top_results = results[:k]

    if not top_results:
        return 0.0

    relevant = sum(
        is_relevant(
            result,
            keywords,
        )
        for result in top_results
    )

    return relevant / len(top_results)


# =========================================================
# EVALUATION
# =========================================================


def run_precision_evals():

    print()
    print("=" * 60)
    print("RETRIEVAL PRECISION@K EVALUATION")
    print("=" * 60)

    all_scores = []

    for index, case in enumerate(
        TEST_CASES,
        start=1,
    ):

        agent = case["agent"]
        query = case["query"]
        keywords = case["keywords"]

        print()
        print(
            f"[{index}] {agent.upper()}"
        )

        print(
            f"Query: {query}"
        )

        try:

            search_function = (
                SEARCH_FUNCTIONS[agent]
            )

            results = search_function(
                query
            )

            if not results:

                print(
                    "No results returned."
                )

                all_scores.append(
                    0.0
                )

                continue

            print(
                f"Results returned: "
                f"{len(results)}"
            )

            # -----------------------------------------
            # Show relevance for each result
            # -----------------------------------------

            for position, result in enumerate(
                results[:5],
                start=1,
            ):

                relevant = is_relevant(
                    result,
                    keywords,
                )

                status = (
                    "RELEVANT"
                    if relevant
                    else "IRRELEVANT"
                )

                title = (
                    result.get("title")
                    or result.get("full_name")
                    or result.get("name")
                    or "Unknown"
                )

                print(
                    f"  #{position} "
                    f"{status}: "
                    f"{title}"
                )

            score = precision_at_k(
                results,
                keywords,
                k=5,
            )

            all_scores.append(
                score
            )

            print(
                f"Precision@5: "
                f"{score:.2f}"
            )

        except Exception as exc:

            print(
                "ERROR:",
                exc,
            )

            all_scores.append(
                0.0
            )

    # =====================================================
    # SUMMARY
    # =====================================================

    total = len(
        all_scores
    )

    average = (
        sum(all_scores) / total
        if total
        else 0.0
    )

    good_queries = sum(
        score >= 0.8
        for score in all_scores
    )

    print()
    print("=" * 60)
    print("PRECISION@K SUMMARY")
    print("=" * 60)

    print(
        f"Tests:             {total}"
    )

    print(
        f"Precision >= 0.8: "
        f"{good_queries}/{total}"
    )

    print(
        f"Average P@5:       "
        f"{average:.2f}"
    )


if __name__ == "__main__":

    run_precision_evals()