import re
from typing import Literal

from pydantic import BaseModel

from app.graph.workflow import graph
from app.llm.models import relevance_llm


# =========================================================
# SCHEMA
# =========================================================

class CitationJudgment(BaseModel):

    verdict: Literal[
        "SUPPORTED",
        "PARTIALLY_SUPPORTED",
        "UNSUPPORTED",
    ]

    reason: str


# =========================================================
# STRUCTURED CITATION JUDGE
# =========================================================

citation_judge_llm = (
    relevance_llm.with_structured_output(
        CitationJudgment
    )
)


# =========================================================
# TEST QUERIES
# =========================================================

TEST_CASES = [

    "Find GitHub implementations of Vision Transformers.",

    "Find research papers about Retrieval-Augmented Generation.",

    "Explain LangGraph persistence.",

    "Explain agent memory architecture.",

    "Research recent developments in multi-agent memory systems.",
]


# =========================================================
# EXTRACT EVIDENCE BLOCK
# =========================================================

def extract_evidence_block(
    merged_context: str,
    evidence_id: str,
) -> str:
    """
    Extract the evidence block belonging to an ID such as:

    W1
    G2
    P1
    M3
    """

    pattern = (
        rf"\[{re.escape(evidence_id)}\]"
        rf"\s*(.*?)"
        rf"(?="
        rf"\n\s*-{{10,}}\s*\n"
        rf"|\n===== "
        rf"|\Z"
        rf")"
    )

    match = re.search(
        pattern,
        merged_context,
        flags=re.DOTALL,
    )

    if not match:
        return ""

    return (
        match.group(1)
        .strip()
    )


# =========================================================
# CITATION JUDGE
# =========================================================

def judge_citation(
    claim: str,
    evidence: str,
) -> CitationJudgment:
    """
    Judge whether cited evidence supports a claim.
    """

    if not evidence.strip():

        return CitationJudgment(
            verdict="UNSUPPORTED",
            reason=(
                "No evidence was supplied "
                "for this claim."
            ),
        )

    prompt = f"""
You are a strict citation correctness evaluator.

Judge whether the EVIDENCE supports the CLAIM.

CLAIM:

{claim}


EVIDENCE:

{evidence}


Use exactly one verdict:

SUPPORTED

Every factual component of the claim is established
by the evidence.


PARTIALLY_SUPPORTED

At least one factual component is established by the
evidence, but another factual component is not
established.


UNSUPPORTED

The central factual proposition is absent,
irrelevant, contradicted, or cannot be established
from the supplied evidence.


IMPORTANT RULES

- Use ONLY the supplied evidence.

- Do not use outside knowledge.

- Semantic equivalence is sufficient.

- Exact wording is NOT required.

- Do not require information that the claim itself
  does not assert.

- Additional information in the evidence does not
  weaken support.

- Do not penalize a claim merely because the evidence
  contains more detail than the claim.

- Do not penalize a claim merely because the claim
  summarizes or paraphrases the evidence.

- Do not infer exclusivity unless the claim itself
  asserts exclusivity.

- Do not infer causation from correlation.

- Do not infer quantitative performance unless the
  exact quantitative information is established by
  the evidence.


PARTIAL SUPPORT RULE

PARTIALLY_SUPPORTED is allowed ONLY when you can
identify a specific factual component of the claim
that is missing from the evidence.

If you return PARTIALLY_SUPPORTED, your reason must
state exactly which factual component is unsupported.


NUMERICAL CLAIM RULE

For numerical or quantitative claims, the evidence
must establish:

1. the relevant number,
2. what that number measures,
3. the relationship asserted by the claim.


FINAL CONSISTENCY CHECK

Before choosing the verdict:

1. Break the claim into factual components.

2. Check every component against the evidence.

3. If every factual component is established:
   SUPPORTED

4. If some factual components are established but
   another specific component is missing:
   PARTIALLY_SUPPORTED

5. If the central factual proposition cannot be
   established:
   UNSUPPORTED


Return the required structured judgment.
"""

    judgment = citation_judge_llm.invoke(
        prompt
    )

    if judgment is None:

        raise RuntimeError(
            "Citation judge returned no result."
        )

    return judgment


# =========================================================
# RUN CITATION EVALUATIONS
# =========================================================

def run_citation_evals():

    print()
    print("=" * 60)
    print("CLAIM-LEVEL CITATION EVALUATION")
    print("=" * 60)

    # =====================================================
    # GLOBAL COUNTERS
    # =====================================================

    total_claims = 0

    cited_claims = 0

    supported = 0

    partially_supported = 0

    unsupported = 0

    missing_evidence = 0

    judge_errors = 0

    graph_errors = 0

    # =====================================================
    # RUN TEST CASES
    # =====================================================

    for test_index, query in enumerate(
        TEST_CASES,
        start=1,
    ):

        print()
        print("=" * 60)
        print(
            f"[{test_index}] {query}"
        )
        print("=" * 60)

        try:

            state = graph.invoke(
                {
                    "query": query
                }
            )

            result = state.get(
                "research_result"
            )

            merged_context = str(
                state.get(
                    "merged_context",
                    "",
                )
                or ""
            )

            # =============================================
            # RESULT CHECK
            # =============================================

            if result is None:

                print(
                    "FAIL: No research_result"
                )

                continue

            findings = (
                getattr(
                    result,
                    "key_findings",
                    [],
                )
                or []
            )

            if not findings:

                print(
                    "No findings generated."
                )

                continue

            # =============================================
            # EVALUATE EACH CLAIM
            # =============================================

            for claim_index, finding in enumerate(
                findings,
                start=1,
            ):

                total_claims += 1

                claim = str(
                    getattr(
                        finding,
                        "claim",
                        "",
                    )
                    or ""
                ).strip()

                evidence_ids = (
                    getattr(
                        finding,
                        "evidence_ids",
                        [],
                    )
                    or []
                )

                print()
                print(
                    f"Claim {claim_index}:"
                )

                print(
                    claim
                )

                print(
                    "Citations:",
                    evidence_ids,
                )

                # =========================================
                # EMPTY CLAIM
                # =========================================

                if not claim:

                    print(
                        "Verdict: UNSUPPORTED"
                    )

                    print(
                        "Reason: Empty claim."
                    )

                    unsupported += 1

                    continue

                # =========================================
                # CITATION COMPLETENESS
                # =========================================

                if not evidence_ids:

                    print(
                        "Verdict: UNSUPPORTED"
                    )

                    print(
                        "Reason: Claim has no citation."
                    )

                    unsupported += 1

                    continue

                cited_claims += 1

                # =========================================
                # COLLECT CITED EVIDENCE
                # =========================================

                evidence_blocks = []

                for evidence_id in evidence_ids:

                    evidence = (
                        extract_evidence_block(
                            merged_context,
                            evidence_id,
                        )
                    )

                    if evidence:

                        evidence_blocks.append(
                            f"[{evidence_id}]\n"
                            f"{evidence}"
                        )

                    else:

                        missing_evidence += 1

                        print(
                            "Missing evidence:",
                            evidence_id,
                        )

                # =========================================
                # NO EVIDENCE FOUND
                # =========================================

                if not evidence_blocks:

                    print(
                        "Verdict: UNSUPPORTED"
                    )

                    print(
                        "Reason: No cited evidence "
                        "could be extracted."
                    )

                    unsupported += 1

                    continue

                combined_evidence = (
                    "\n\n".join(
                        evidence_blocks
                    )
                )

                # =========================================
                # JUDGE CLAIM
                # =========================================

                try:

                    judgment = (
                        judge_citation(
                            claim,
                            combined_evidence,
                        )
                    )

                except Exception as exc:

                    judge_errors += 1

                    print(
                        "JUDGE ERROR:",
                        type(exc).__name__,
                        str(exc),
                    )

                    # Evaluator infrastructure failure
                    # must NOT be counted as a research
                    # system failure.
                    continue

                # =========================================
                # PRINT JUDGMENT
                # =========================================

                print(
                    "Verdict:",
                    judgment.verdict,
                )

                print(
                    "Reason:",
                    judgment.reason,
                )

                # =========================================
                # COUNT JUDGMENT
                # =========================================

                if (
                    judgment.verdict
                    == "SUPPORTED"
                ):

                    supported += 1

                elif (
                    judgment.verdict
                    == "PARTIALLY_SUPPORTED"
                ):

                    partially_supported += 1

                elif (
                    judgment.verdict
                    == "UNSUPPORTED"
                ):

                    unsupported += 1

                else:

                    judge_errors += 1

                    print(
                        "JUDGE ERROR: Unknown verdict:",
                        judgment.verdict,
                    )

        except Exception as exc:

            graph_errors += 1

            print(
                "GRAPH ERROR:",
                type(exc).__name__,
                str(exc),
            )

    # =====================================================
    # SUMMARY CALCULATIONS
    # =====================================================

    judged = (
        supported
        + partially_supported
        + unsupported
    )

    citation_completeness = (
        cited_claims
        / total_claims
        if total_claims
        else 0.0
    )

    judge_success_rate = (
        judged
        / total_claims
        if total_claims
        else 0.0
    )

    citation_correctness = (
        supported
        / judged
        if judged
        else None
    )

    partial_or_better = (
        (
            supported
            + partially_supported
        )
        / judged
        if judged
        else None
    )

    unsupported_rate = (
        unsupported
        / judged
        if judged
        else None
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    print()
    print("=" * 60)
    print("CITATION EVALUATION SUMMARY")
    print("=" * 60)

    print(
        f"Total claims:              "
        f"{total_claims}"
    )

    print(
        f"Claims with citations:     "
        f"{cited_claims}"
    )

    print(
        f"Successfully judged:       "
        f"{judged}/{total_claims}"
    )

    print(
        f"Supported:                 "
        f"{supported}"
    )

    print(
        f"Partially supported:       "
        f"{partially_supported}"
    )

    print(
        f"Unsupported:               "
        f"{unsupported}"
    )

    print(
        f"Missing evidence blocks:   "
        f"{missing_evidence}"
    )

    print(
        f"Judge errors:              "
        f"{judge_errors}"
    )

    print(
        f"Graph errors:              "
        f"{graph_errors}"
    )

    print(
        f"Judge success rate:        "
        f"{judge_success_rate:.2%}"
    )

    print(
        f"Citation completeness:     "
        f"{citation_completeness:.2%}"
    )

    # =====================================================
    # ONLY REPORT JUDGE METRICS IF JUDGE ACTUALLY WORKED
    # =====================================================

    if citation_correctness is not None:

        print(
            f"Citation correctness:      "
            f"{citation_correctness:.2%}"
        )

    else:

        print(
            "Citation correctness:      "
            "N/A (citation judge failed)"
        )

    if partial_or_better is not None:

        print(
            f"Partial-or-better rate:    "
            f"{partial_or_better:.2%}"
        )

    else:

        print(
            "Partial-or-better rate:    "
            "N/A"
        )

    if unsupported_rate is not None:

        print(
            f"Unsupported claim rate:    "
            f"{unsupported_rate:.2%}"
        )

    else:

        print(
            "Unsupported claim rate:    "
            "N/A"
        )

    # =====================================================
    # HEALTH CHECK
    # =====================================================

    print()
    print("-" * 60)
    print("EVALUATOR HEALTH")
    print("-" * 60)

    if judge_errors == 0:

        print(
            "Citation judge:            HEALTHY"
        )

    else:

        print(
            "Citation judge:            DEGRADED"
        )

    if graph_errors == 0:

        print(
            "Research graph:            HEALTHY"
        )

    else:

        print(
            "Research graph:            DEGRADED"
        )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    run_citation_evals()