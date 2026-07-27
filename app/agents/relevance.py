import json
import time
from requests.exceptions import ReadTimeout

from app.llm.models import relevance_llm
from app.schemas.relevance import RelevanceResult


def evaluate_relevance(
    user_query: str,
    content: str,
) -> RelevanceResult:

    prompt = f"""
You are a relevance evaluator.

USER QUERY:
{user_query}

RESULT:
{content[:4000]}

Return ONLY valid JSON:

{{
    "relevant": true,
    "score": 0.0,
    "reason": "brief reason"
}}

Score from 0.0 to 1.0.
A result is relevant only when it directly helps answer the query.
"""

    # One retry only
    for attempt in range(2):
        try:
            response = relevance_llm.invoke(prompt)
            raw = response.content.strip()

            if raw.startswith("```"):
                raw = raw.removeprefix("```json")
                raw = raw.removeprefix("```")
                raw = raw.removesuffix("```").strip()

            data = json.loads(raw)

            return RelevanceResult.model_validate(data)

        except ReadTimeout:
            print(
                f"[RELEVANCE] NVIDIA timeout "
                f"(attempt {attempt + 1}/2)"
            )

            if attempt == 0:
                time.sleep(1)

        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[RELEVANCE] Invalid model output: {exc}")
            break

        except Exception as exc:
            print(f"[RELEVANCE] Evaluation failed: {exc}")
            break

    # Fail open: don't throw away potentially useful evidence
    return RelevanceResult(
        relevant=True,
        score=0.5,
        reason="Relevance evaluation unavailable."
    )