from app.graph.nodes import planner_node

from app.evals.datasets.planner_cases import (
    PLANNER_CASES
)


def get_agents(plan):
    """
    Extract agent names from planner output.
    """

    return {
        step.agent
        for step in plan
    }


def calculate_metrics(
    expected: set,
    actual: set,
):
    """
    Calculate routing precision, recall and F1.
    """

    true_positive = len(
        expected & actual
    )

    false_positive = len(
        actual - expected
    )

    false_negative = len(
        expected - actual
    )

    precision = (
        true_positive
        / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )

    recall = (
        true_positive
        / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    exact_match = (
        expected == actual
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match,
    }


def run_planner_evals():

    print()
    print("=" * 60)
    print("PLANNER ROUTING EVALUATION")
    print("=" * 60)

    results = []

    for i, case in enumerate(
        PLANNER_CASES,
        start=1,
    ):

        query = case["query"]
        expected = case["expected_agents"]

        print()
        print(f"[{i}] {query}")

        try:

            output = planner_node(
                {
                    "query": query
                }
            )

            plan = output.get(
                "plan",
                [],
            )

            actual = get_agents(
                plan
            )

            metrics = calculate_metrics(
                expected,
                actual,
            )

            print(
                "Expected:",
                sorted(expected),
            )

            print(
                "Actual:  ",
                sorted(actual),
            )

            print(
                f"Precision: "
                f"{metrics['precision']:.2f}"
            )

            print(
                f"Recall:    "
                f"{metrics['recall']:.2f}"
            )

            print(
                f"F1:        "
                f"{metrics['f1']:.2f}"
            )

            print(
                "Exact:",
                metrics["exact_match"],
            )

            results.append(
                metrics
            )

        except Exception as exc:

            print(
                "ERROR:",
                exc,
            )

            results.append(
                {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "exact_match": False,
                }
            )

    # ==========================================
    # SUMMARY
    # ==========================================

    total = len(results)

    avg_precision = sum(
        r["precision"]
        for r in results
    ) / total

    avg_recall = sum(
        r["recall"]
        for r in results
    ) / total

    avg_f1 = sum(
        r["f1"]
        for r in results
    ) / total

    exact_matches = sum(
        r["exact_match"]
        for r in results
    )

    print()
    print("=" * 60)
    print("PLANNER EVALUATION SUMMARY")
    print("=" * 60)

    print(
        f"Tests:             {total}"
    )

    print(
        f"Exact Match:       "
        f"{exact_matches}/{total}"
    )

    print(
        f"Routing Accuracy:  "
        f"{exact_matches / total * 100:.1f}%"
    )

    print(
        f"Avg Precision:     "
        f"{avg_precision:.2f}"
    )

    print(
        f"Avg Recall:        "
        f"{avg_recall:.2f}"
    )

    print(
        f"Avg F1:            "
        f"{avg_f1:.2f}"
    )


if __name__ == "__main__":
    run_planner_evals()