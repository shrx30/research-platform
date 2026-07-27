from typing import Literal

from pydantic import BaseModel


class ResearchResult(BaseModel):

    summary: str

    key_findings: list[str]

    sources_used: list[str]

    missing_information: list[str] 

    confidence: Literal["Low", "Medium", "High"]    
        