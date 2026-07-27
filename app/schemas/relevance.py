from pydantic import BaseModel, Field


class RelevanceResult(BaseModel):
    relevant: bool

    score: float = Field(
        ge=0.0,
        le=1.0
    )

    reason: str