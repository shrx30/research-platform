from app.graph.state import ResearchState


def route_agents(state: ResearchState):

    routes = []

    for step in state["plan"]:
        routes.append(step.agent)

    return routes