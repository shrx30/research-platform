from dataclasses import dataclass


MAX_QUERY_LENGTH = 4000


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""


def validate_input(
    query: str,
) -> GuardResult:

    if not query:
        return GuardResult(
            allowed=False,
            reason="Research question is empty.",
        )

    query = query.strip()

    if not query:
        return GuardResult(
            allowed=False,
            reason="Research question is empty.",
        )

    if len(query) > MAX_QUERY_LENGTH:
        return GuardResult(
            allowed=False,
            reason=(
                "Research question exceeds "
                f"{MAX_QUERY_LENGTH} characters."
            ),
        )

    return GuardResult(
        allowed=True
    )


def validate_report(
    report: str,
) -> GuardResult:

    if not report:
        return GuardResult(
            allowed=False,
            reason="Research report is empty.",
        )

    suspicious_output = (
        "you are chatgpt",
        "system message provided",
        "hidden system prompt",
        "knowledge cutoff:",
    )

    lowered = report.lower()

    for pattern in suspicious_output:

        if pattern in lowered:

            return GuardResult(
                allowed=False,
                reason=(
                    "The generated report appears to "
                    "contain internal model instructions."
                ),
            )

    return GuardResult(
        allowed=True
    )