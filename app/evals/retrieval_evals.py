from app.agents.web import run as web_run
from app.agents.github import run as github_run
from app.agents.papers import run as papers_run

from app.evals.datasets.retrieval_cases import (
    RETRIEVAL_CASES,
)


AGENTS = {
    "web": web_run,
    "github": github_run,
    "papers": papers_run,
}


def keyword_relevance(
    text: str,
    keywords: list[str],
) -> float:

    if not text:
        return 0.0

    text = text.lower()

    hits = 0

    for keyword in keywords:

        if keyword.lower() in text:
            hits += 1

    return hits / len(keywords)


def run_retrieval_evals():

    print()
    print("=" * 60)
    print("RETRIEVAL RELEVANCE EVALUATION")
    print("=" * 60)

    scores = []

    for i, case in enumerate(
        RETRIEVAL_CASES,
        start=1,
    ):

        agent_name = case["agent"]
        query = case["query"]
        keywords = case["keywords"]

        print()
        print(
            f"[{i}] {agent_name.upper()}"
        )

        print(
            f"Query: {query}"
        )

        try:

            agent = AGENTS[
                agent_name
            ]

            result = agent(
                query
            )

            score = keyword_relevance(
                result,
                keywords,
            )

            scores.append(
                score
            )

            print(
                "Expected keywords:",
                keywords,
            )

            print(
                f"Result length: "
                f"{len(result or '')}"
            )

            print(
                f"Relevance: "
                f"{score:.2f}"
            )

            if score >= 0.5:
                print("PASS")
            else:
                print("FAIL")

        except Exception as exc:

            print(
                "ERROR:",
                exc,
            )

            scores.append(
                0.0
            )

    # ======================================
    # SUMMARY
    # ======================================

    total = len(scores)

    passed = sum(
        score >= 0.5
        for score in scores
    )

    average = (
        sum(scores) / total
        if total
        else 0
    )

    print()
    print("=" * 60)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)

    print(
        f"Tests:              {total}"
    )

    print(
        f"Passed:             "
        f"{passed}/{total}"
    )

    print(
        f"Pass Rate:          "
        f"{passed / total * 100:.1f}%"
    )

    print(
        f"Average Relevance:  "
        f"{average:.2f}"
    )


if __name__ == "__main__":
    run_retrieval_evals()