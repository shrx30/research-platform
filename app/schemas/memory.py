from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    topic: str
    content: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class MemoryWriteResult(BaseModel):
    memories: list[MemoryItem]