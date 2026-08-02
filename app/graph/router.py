from app.graph.state import ResearchState


VALID_AGENTS = {
    "web",
    "github",
    "papers",
    "memory",
}


def route_agents(state: ResearchState):
    """
    Route planner output to selected research agents.
    """

    plan = state.get(
        "plan",
        [],
    )

    routes = []

    for step in plan:

        agent = getattr(
            step,
            "agent",
            None,
        )

        if (
            agent in VALID_AGENTS
            and agent not in routes
        ):
            routes.append(agent)

    # Safety fallback
    if not routes:
        print(
            "[ROUTER] No valid agents selected. "
            "Using web."
        )

        return ["web"]

    print(
        "[ROUTER] Selected:",
        routes,
    )

    return routes