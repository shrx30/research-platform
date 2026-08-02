import re

from app.graph.workflow import graph


TEST_CASES = [
    "Find GitHub implementations of Vision Transformers.",
    "Find research papers about Retrieval-Augmented Generation.",
    "Explain LangGraph persistence.",
    "Explain agent memory architecture.",
    "Research multi-agent memory systems.",
]


# =========================================================
# TEXT HELPERS
# =========================================================

def normalize(text: str) -> str:

    text = str(text).lower()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def important_words(text: str) -> set[str]:
    """
    Extract useful words from a claim.
    """

    stopwords = {
        "the", "a", "an", "and", "or",
        "of", "to", "in", "for", "on",
        "with", "is", "are", "was", "were",
        "that", "this", "it", "as", "by",
        "from", "be", "can", "using",
        "used", "use",
    }

    words = normalize(text).split()

    return {
        word
        for word in words
        if (
            word not in stopwords
            and len(word) > 2
        )
    }


# =========================================================
# CLAIM GROUNDEDNESS
# =========================================================

def claim_groundedness(
    claim: str,
    context: str,
) -> float:
    """
    Simple lexical grounding score.

    Measures how many important words from
    the generated claim occur in retrieved evidence.
    """

    claim_words = important_words(
        claim
    )

    if not claim_words:
        return 0.0

    context_words = set(
        normalize(context).split()
    )

    supported = (
        claim_words
        & context_words
    )

    return (
        len(supported)
        / len(claim_words)
    )


# =========================================================
# EVALUATION
# =========================================================

def run_groundedness_evals():

    print()
    print("=" * 60)
    print("GROUNDEDNESS EVALUATION")
    print("=" * 60)

    all_scores = []

    for index, query in enumerate(
        TEST_CASES,
        start=1,
    ):

        print()
        print(
            f"[{index}] {query}"
        )

        try:

            state = graph.invoke(
                {
                    "query": query
                }
            )

            result = state.get(
                "research_result"
            )

            context = state.get(
                "merged_context",
                "",
            )

            if result is None:

                print(
                    "FAIL: No research result."
                )

                continue

            # =========================================
            # COLLECT CLAIMS
            # =========================================

            claims = []

            summary = getattr(
                result,
                "summary",
                "",
            )

            if summary:

                # Break summary into sentences

                summary_claims = re.split(
                    r"(?<=[.!?])\s+",
                    summary,
                )

                claims.extend(
                    claim
                    for claim in summary_claims
                    if claim.strip()
                )

            findings = (
                getattr(
                    result,
                    "key_findings",
                    [],
                )
                or []
            )

            claims.extend(
                findings
            )

            # =========================================
            # SCORE CLAIMS
            # =========================================

            claim_scores = []

            for claim in claims:

                score = claim_groundedness(
                    claim,
                    context,
                )

                claim_scores.append(
                    score
                )

                status = (
                    "PASS"
                    if score >= 0.60
                    else "FAIL"
                )

                print()
                print(
                    f"Claim: {claim}"
                )

                print(
                    f"Groundedness: "
                    f"{score:.2f} "
                    f"{status}"
                )

            # =========================================
            # QUERY SCORE
            # =========================================

            query_score = (
                sum(claim_scores)
                / len(claim_scores)
                if claim_scores
                else 0.0
            )

            all_scores.extend(
                claim_scores
            )

            print()
            print(
                f"Query Groundedness: "
                f"{query_score:.2f}"
            )

        except Exception as exc:

            print(
                "ERROR:",
                exc,
            )

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    average = (
        sum(all_scores)
        / len(all_scores)
        if all_scores
        else 0.0
    )

    grounded = sum(
        score >= 0.60
        for score in all_scores
    )

    total = len(
        all_scores
    )

    print()
    print("=" * 60)
    print(
        "GROUNDEDNESS SUMMARY"
    )
    print("=" * 60)

    print(
        f"Claims evaluated:       {total}"
    )

    print(
        f"Grounded claims:        "
        f"{grounded}/{total}"
    )

    print(
        f"Grounded claim rate:    "
        f"{grounded / total:.2%}"
        if total
        else
        "Grounded claim rate:    0.00%"
    )

    print(
        f"Average groundedness:   "
        f"{average:.2f}"
    )


if __name__ == "__main__":

    run_groundedness_evals()