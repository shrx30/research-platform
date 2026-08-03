import json
import re
from typing import Any

from app.llm.models import structured_base
from app.schemas.memory import MemoryWriteResult
from app.Tools.memory_tools import store_memory


# =========================================================
# RESPONSE HELPERS
# =========================================================


def _response_text(
    response: Any,
) -> str:
    """
    Extract plain text from a LangChain response.
    """

    if response is None:
        return ""

    content = getattr(
        response,
        "content",
        response,
    )

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        parts: list[str] = []

        for item in content:

            if isinstance(item, str):

                parts.append(
                    item
                )

            elif isinstance(item, dict):

                text = item.get(
                    "text"
                )

                if text:

                    parts.append(
                        str(text)
                    )

        return "\n".join(
            parts
        ).strip()

    return str(
        content
    ).strip()


def _extract_json(
    text: str,
) -> dict:
    """
    Extract a JSON object from normal model output.

    Handles:
    - raw JSON
    - ```json fences
    - surrounding text
    """

    if not text:

        raise ValueError(
            "Memory writer returned empty output."
        )

    text = text.strip()

    # Remove opening code fence.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove closing code fence.
    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    # Try normal JSON first.
    try:

        data = json.loads(
            text
        )

        if not isinstance(
            data,
            dict,
        ):

            raise ValueError(
                "Memory writer JSON must "
                "be an object."
            )

        return data

    except json.JSONDecodeError:

        pass

    # Try extracting the outermost object.
    start = text.find(
        "{"
    )

    end = text.rfind(
        "}"
    )

    if (
        start == -1
        or end == -1
        or end <= start
    ):

        raise ValueError(
            "No JSON object found in "
            "memory writer response."
        )

    data = json.loads(
        text[start:end + 1]
    )

    if not isinstance(
        data,
        dict,
    ):

        raise ValueError(
            "Memory writer JSON must "
            "be an object."
        )

    return data


# =========================================================
# MEMORY WRITER
# =========================================================


def write_memories(
    user_query: str,
    research_result,
) -> int:
    """
    Extract reusable findings from a completed research
    result and store them in long-term memory.

    Uses normal JSON generation rather than
    with_structured_output(), avoiding NVIDIA guided_json.
    """

    # =====================================================
    # SERIALIZE RESEARCH RESULT
    # =====================================================

    if hasattr(
        research_result,
        "model_dump_json",
    ):

        research_text = (
            research_result.model_dump_json(
                indent=2
            )
        )

    elif hasattr(
        research_result,
        "model_dump",
    ):

        research_text = json.dumps(
            research_result.model_dump(),
            indent=2,
            default=str,
        )

    else:

        research_text = str(
            research_result
        )

    # =====================================================
    # PROMPT
    # =====================================================

    prompt = f"""
You are the long-term memory extraction component of a
research system.

Extract reusable factual research findings from the supplied
research result.


USER QUERY

{user_query}


RESEARCH RESULT

{research_text}


MEMORY RULES

1. Store factual research findings only.

2. Do not store execution details.

3. Do not store planner decisions.

4. Do not store search queries.

5. Do not store debugging information.

6. Do not store latency or evaluation information.

7. Each memory must contain exactly one clear reusable finding.

8. Every memory must make sense independently without requiring
   the original conversation.

9. Do not invent facts.

10. Do not add information that is absent from the research
    result.

11. Avoid duplicate or substantially overlapping memories.

12. Prefer findings that are likely to be useful in future
    research.

13. Extract at most 5 memories.

14. If there are no useful findings, return an empty memories
    list.

15. Sources must come ONLY from sources present in the research
    result.

16. Do not invent or reconstruct source URLs.


For every memory return:

topic:
A short descriptive topic.

content:
One self-contained factual finding.

sources:
A list of supporting sources from the research result.

confidence:
One of:
Low
Medium
High


Return ONLY valid JSON in exactly this structure:

{{
    "memories": [
        {{
            "topic": "Short topic",
            "content": "Self-contained factual finding.",
            "sources": [
                "https://example.com/source"
            ],
            "confidence": "High"
        }}
    ]
}}

If there are no memories worth storing, return:

{{
    "memories": []
}}

Do not use Markdown.
Do not use code fences.
Do not include explanations.
Return only the JSON object.
"""

    # =====================================================
    # NORMAL LLM CALL
    # =====================================================

    response = structured_base.invoke(
        prompt
    )

    content = _response_text(
        response
    )

    if not content:

        raise RuntimeError(
            "Memory writer model returned "
            "empty output."
        )

    # =====================================================
    # JSON PARSING
    # =====================================================

    data = _extract_json(
        content
    )

    # Pydantic still validates the output.
    # We simply aren't asking NVIDIA to perform
    # guided structured generation.
    extracted = (
        MemoryWriteResult.model_validate(
            data
        )
    )

    # =====================================================
    # SAFETY LIMIT
    # =====================================================

    memories = extracted.memories[
        :5
    ]

    if not memories:

        print(
            "[MEMORY WRITE] "
            "No reusable memories extracted."
        )

        return 0

    # =====================================================
    # STORE MEMORIES
    # =====================================================

    stored = 0

    for memory in memories:

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

            # One failed vector DB write should not
            # prevent other memories from being stored.
            print(
                "[MEMORY WRITE] "
                f"Failed to store "
                f"'{memory.topic}': "
                f"{exc}"
            )

    return stored