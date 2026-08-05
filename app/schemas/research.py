from typing import Literal

from pydantic import BaseModel, Field


# =========================================================
# CLAIM
# =========================================================

class ResearchClaim(BaseModel):
    """
    A research finding together with the evidence
    supporting it.
    """

    claim: str

    evidence_ids: list[str] = Field(
        default_factory=list
    )


# =========================================================
# RESEARCH RESULT
# =========================================================

class ResearchResult(BaseModel):

    summary: str

    key_findings: list[ResearchClaim] = Field(
        default_factory=list
    )

    sources_used: list[str] = Field(
        default_factory=list
    )

    missing_information: list[str] = Field(
        default_factory=list
    )

    confidence: Literal[
        "Low",
        "Medium",
        "High",
    ]