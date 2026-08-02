EVAL_CASES = [
    {
        "query": "Find GitHub implementations of Vision Transformers.",
        "expected_agents": ["github"],
        "expected_terms": [
            "vision transformer",
            "github",
        ],
    },

    {
        "query": "Find research papers about agent memory architectures.",
        "expected_agents": ["papers"],
        "expected_terms": [
            "memory",
            "agent",
        ],
    },

    {
        "query": "Explain LangGraph persistence and find its documentation.",
        "expected_agents": ["web"],
        "expected_terms": [
            "langgraph",
            "persistence",
        ],
    },

    {
        "query": (
            "Research agent memory and give me "
            "papers and GitHub implementations."
        ),
        "expected_agents": [
            "web",
            "github",
            "papers",
        ],
        "expected_terms": [
            "memory",
            "agent",
        ],
    },
]