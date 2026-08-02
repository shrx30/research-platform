import re

from app.graph.workflow import graph


# =========================================================
# TEST CASES
# =========================================================

TEST_CASES = [
    "Find GitHub implementations of Vision Transformers.",
    "Find research papers about Retrieval-Augmented Generation.",
    "Explain LangGraph persistence.",
    "Explain agent memory architecture.",
    "Research multi-agent memory systems.",
]


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_text(text: str) -> str:
    """
    Normalize text so small formatting differences
    do not cause source validation to fail.
    """

    text = str(text).lower().strip()

    # Normalize different dash characters
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    # Remove punctuation
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
    )

    # Collapse multiple spaces
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# =========================================================
# PAPER TITLE EXTRACTION
# =========================================================

def extract_source_title(source: str) -> str:
    """
    Convert:

    Author et al. (2025) - Paper Title

    into:

    Paper Title
    """

    source = str(source).strip()

    parts = re.split(
        r"\s+[-–—]\s+",
        source,
        maxsplit=1,
    )

    if len(parts) == 2:
        return parts[1].strip()

    return source


# =========================================================
# SOURCE VALIDITY
# =========================================================

def calculate_source_validity(
    sources: list[str],
    merged_context: str,
):
    """
    Validate sources against retrieved evidence.

    Supports:
    - GitHub URLs
    - web URLs
    - paper citations
    - paper titles
    """

    if not sources:
        return {
            "score": 0.0,
            "valid": [],
            "invalid": [],
            "total": 0,
        }

    raw_context = (
        str(merged_context)
        .lower()
        .strip()
    )

    normalized_context = normalize_text(
        merged_context
    )

    valid = []
    invalid = []

    for source in sources:

        source_string = str(source).strip()

        raw_source = (
            source_string
            .lower()
            .rstrip("/")
        )

        # =============================================
        # 1. DIRECT MATCH
        #
        # Works well for URLs.
        # =============================================

        if (
            raw_source
            and raw_source in raw_context
        ):
            valid.append(source)
            continue

        # =============================================
        # 2. PAPER TITLE MATCH
        #
        # Example:
        #
        # Author et al. (2025) - Paper Title
        #
        # becomes:
        #
        # Paper Title
        # =============================================

        title = extract_source_title(
            source_string
        )

        normalized_title = normalize_text(
            title
        )

        if (
            len(normalized_title) >= 10
            and normalized_title
            in normalized_context
        ):
            valid.append(source)
            continue

        # =============================================
        # SOURCE COULD NOT BE VERIFIED
        # =============================================

        invalid.append(source)

    score = (
        len(valid) / len(sources)
    )

    return {
        "score": score,
        "valid": valid,
        "invalid": invalid,
        "total": len(sources),
    }


# =========================================================
# EVALUATION
# =========================================================

def run_source_validity_evals():

    print()
    print("=" * 60)
    print("SOURCE VALIDITY EVALUATION")
    print("=" * 60)

    scores = []

    for index, query in enumerate(
        TEST_CASES,
        start=1,
    ):

        print()
        print(
            f"[{index}] {query}"
        )

        try:

            # =========================================
            # RUN FULL RESEARCH GRAPH
            # =========================================

            state = graph.invoke(
                {
                    "query": query
                }
            )

            result = state.get(
                "research_result"
            )

            merged_context = state.get(
                "merged_context",
                "",
            )

            # =========================================
            # CHECK RESULT EXISTS
            # =========================================

            if result is None:

                print(
                    "FAIL: No research_result"
                )

                scores.append(
                    0.0
                )

                continue

            # =========================================
            # GET REPORTED SOURCES
            # =========================================

            sources = (
                getattr(
                    result,
                    "sources_used",
                    [],
                )
                or []
            )

            # =========================================
            # VALIDATE SOURCES
            # =========================================

            evaluation = (
                calculate_source_validity(
                    sources,
                    merged_context,
                )
            )

            score = evaluation[
                "score"
            ]

            scores.append(
                score
            )

            # =========================================
            # DISPLAY RESULT
            # =========================================

            print(
                f"Sources reported: "
                f"{evaluation['total']}"
            )

            print(
                f"Valid sources:    "
                f"{len(evaluation['valid'])}"
            )

            print(
                f"Invalid sources:  "
                f"{len(evaluation['invalid'])}"
            )

            print(
                f"Source validity:  "
                f"{score:.2f}"
            )

            # =========================================
            # SHOW INVALID SOURCES
            # =========================================

            if evaluation[
                "invalid"
            ]:

                print()
                print(
                    "INVALID SOURCES:"
                )

                for source in evaluation[
                    "invalid"
                ]:

                    print(
                        f"  - {source}"
                    )

            # =========================================
            # PASS / FAIL
            # =========================================

            if score == 1.0:

                print(
                    "PASS"
                )

            else:

                print(
                    "FAIL"
                )

        except Exception as exc:

            print(
                "ERROR:",
                exc,
            )

            scores.append(
                0.0
            )

    # =====================================================
    # SUMMARY
    # =====================================================

    total = len(
        scores
    )

    perfect = sum(
        score == 1.0
        for score in scores
    )

    average = (
        sum(scores) / total
        if total
        else 0.0
    )

    unsupported_rate = (
        1.0 - average
    )

    print()
    print("=" * 60)
    print(
        "SOURCE VALIDITY SUMMARY"
    )
    print("=" * 60)

    print(
        f"Tests:                    "
        f"{total}"
    )

    print(
        f"Perfect:                  "
        f"{perfect}/{total}"
    )

    print(
        f"Average Validity:         "
        f"{average:.2f}"
    )

    print(
        f"Unsupported Source Rate:  "
        f"{unsupported_rate:.2%}"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    run_source_validity_evals()