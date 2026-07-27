from typing import Literal
from pydantic import BaseModel


class PlanStep(BaseModel):
    agent: Literal["web", "github", "papers", "memory"]

    # What the agent should accomplish
    task: str

    # Search-engine/API-friendly query
    query: str


class ExecutionPlan(BaseModel):
    steps: list[PlanStep]