PLANNER_CASES = [
    {
        "query": "Find GitHub implementations of Vision Transformers.",
        "expected_agents": {"github"},
    },
    {
        "query": "Find research papers about Retrieval-Augmented Generation.",
        "expected_agents": {"papers"},
    },
    {
        "query": "Find the official LangGraph documentation about persistence.",
        "expected_agents": {"web"},
    },
    {
        "query": "Find papers and open-source implementations of multi-agent memory.",
        "expected_agents": {"papers", "github"},
    },
    {
        "query": "Research recent developments in AI agent memory and relevant academic papers.",
        "expected_agents": {"web", "papers"},
    },
    {
        "query": "Find open-source RAG frameworks and explain their current features.",
        "expected_agents": {"github", "web"},
    },
]