from langgraph.graph import StateGraph, START, END

from app.graph.state import ResearchState
from app.graph.router import route_agents

from app.graph.nodes import (
    planner_node,
    web_agent,
    github_agent,
    paper_agent,
    memory_agent,
    merge_node,
    research_node,
    report_node,
    memory_write_node,
)


# =========================================================
# GRAPH
# =========================================================

builder = StateGraph(ResearchState)


# =========================================================
# NODES
# =========================================================

builder.add_node(
    "planner",
    planner_node,
)

builder.add_node(
    "web",
    web_agent,
)

builder.add_node(
    "github",
    github_agent,
)

builder.add_node(
    "papers",
    paper_agent,
)

builder.add_node(
    "memory",
    memory_agent,
)

builder.add_node(
    "merge",
    merge_node,
)

builder.add_node(
    "research",
    research_node,
)

builder.add_node(
    "report",
    report_node,
)

builder.add_node(
    "memory_write",
    memory_write_node,
)


# =========================================================
# START → PLANNER
# =========================================================

builder.add_edge(
    START,
    "planner",
)


# =========================================================
# PLANNER → SELECTED AGENTS
# =========================================================

builder.add_conditional_edges(
    "planner",
    route_agents,
)


# =========================================================
# AGENTS → MERGE
# =========================================================

builder.add_edge(
    "web",
    "merge",
)

builder.add_edge(
    "github",
    "merge",
)

builder.add_edge(
    "papers",
    "merge",
)

builder.add_edge(
    "memory",
    "merge",
)


# =========================================================
# MERGE → SYNTHESIS
# =========================================================

builder.add_edge(
    "merge",
    "research",
)


# =========================================================
# SYNTHESIS → POST PROCESSING
# =========================================================

builder.add_edge(
    "research",
    "report",
)

builder.add_edge(
    "research",
    "memory_write",
)


# =========================================================
# POST PROCESSING → END
# =========================================================

builder.add_edge(
    "report",
    END,
)

builder.add_edge(
    "memory_write",
    END,
)


# =========================================================
# COMPILE
# =========================================================

graph = builder.compile()
