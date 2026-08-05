import time
import statistics

from app.graph.workflow import graph


TEST_CASES = [
    "Find GitHub implementations of Vision Transformers.",
    "Find research papers about Retrieval-Augmented Generation.",
    "Explain LangGraph persistence.",
    "Research recent developments in multi-agent memory systems.",
    "Find open-source RAG frameworks and explain their current features.",
]


def run_latency_evals():

    print()
    print("=" * 60)
    print("END-TO-END LATENCY EVALUATION")
    print("=" * 60)

    latencies = []
    successful = 0

    for index, query in enumerate(
        TEST_CASES,
        start=1,
    ):

        print()
        print(f"[{index}] {query}")

        start = time.perf_counter()

        try:

            state = graph.invoke(
                {
                    "query": query
                }
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            latencies.append(
                elapsed
            )

            successful += 1

            print(
                f"[E2E LATENCY] "
                f"{elapsed:.2f}s"
            )

            print(
                "Report generated:",
                bool(
                    state.get("report")
                ),
            )

        except Exception as exc:

            elapsed = (
                time.perf_counter()
                - start
            )

            print(
                f"[FAILED AFTER] "
                f"{elapsed:.2f}s"
            )

            print(
                "ERROR:",
                exc,
            )

    # =====================================================
    # SUMMARY
    # =====================================================

    print()
    print("=" * 60)
    print("LATENCY SUMMARY")
    print("=" * 60)

    print(
        f"Tests:               "
        f"{len(TEST_CASES)}"
    )

    print(
        f"Successful:          "
        f"{successful}/{len(TEST_CASES)}"
    )

    if not latencies:
        return

    average = statistics.mean(
        latencies
    )

    median = statistics.median(
        latencies
    )

    minimum = min(
        latencies
    )

    maximum = max(
        latencies
    )

    print(
        f"Average latency:     "
        f"{average:.2f}s"
    )

    print(
        f"Median latency:      "
        f"{median:.2f}s"
    )

    print(
        f"Minimum latency:     "
        f"{minimum:.2f}s"
    )

    print(
        f"Maximum latency:     "
        f"{maximum:.2f}s"
    )


if __name__ == "__main__":

    run_latency_evals()