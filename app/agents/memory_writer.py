from app.llm.models import kimi_k3
from app.schemas.memory import MemoryWriteResult
from app.Tools.memory_tools import store_memory


# =========================================================
# MEMORY WRITER
# =========================================================

memory_writer_llm = kimi_k3.with_structured_output(
    MemoryWriteResult
)


def write_memories(
    user_query: str,
    research_result,
) -> int:

    prompt = f"""
You are the long-term memory writer for a research agent.

USER QUERY:
{user_query}

RESEARCH RESULT:
{research_result}

Extract reusable research findings worth remembering for
future research tasks.

Rules:

1. Store factual findings, not execution details.
2. Each memory should contain one clear finding.
3. Do not store the entire report.
4. Do not store planner steps, search queries, relevance
   scores, debugging information, or agent execution details.
5. Avoid duplicate findings.
6. Prefer findings supported by reliable sources.
7. Never invent facts.
8. Each finding must make sense independently.
9. Extract at most 5 memories.
10. Return an empty list if nothing is worth remembering.

For every memory provide:

- topic
- content
- sources
- confidence

Confidence MUST be a number between 0 and 1.

Examples:

0.95 = strongly supported
0.80 = well supported
0.60 = moderately supported
0.40 = weakly supported

Do not return "High", "Medium", or "Low" for confidence.
Return a numeric value.
"""

    try:

        extracted: MemoryWriteResult = (
            memory_writer_llm.invoke(prompt)
        )

    except Exception as exc:

        print(
            f"[MEMORY WRITE] LLM extraction failed: {exc}"
        )

        raise

    stored = 0

    for memory in extracted.memories:

        try:

            store_memory(
                content=memory.content,
                topic=memory.topic,
                sources=memory.sources,
                confidence=memory.confidence,
            )

            stored += 1

            print(
                f"[MEMORY WRITE] "
                f"{memory.topic}: "
                f"{memory.content[:80]}"
            )

        except Exception as exc:

            print(
                f"[MEMORY WRITE] "
                f"Failed to store memory: {exc}"
            )

    print(
        f"[MEMORY WRITE] Stored {stored} memories."
    )

    return stored
