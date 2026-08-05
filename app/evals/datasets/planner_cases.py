PLANNER_CASES = [

    # =====================================================
    # SINGLE AGENT — GITHUB
    # =====================================================

    {
        "query": "Find GitHub implementations of Vision Transformers.",
        "expected_agents": {"github"},
    },

    {
        "query": "Show me open-source implementations of GraphRAG.",
        "expected_agents": {"github"},
    },

    {
        "query": "Find GitHub repositories implementing AI agent memory.",
        "expected_agents": {"github"},
    },

    # =====================================================
    # SINGLE AGENT — PAPERS
    # =====================================================

    {
        "query": "Find research papers about Retrieval-Augmented Generation.",
        "expected_agents": {"papers"},
    },

    {
        "query": "Find academic papers on transformer interpretability.",
        "expected_agents": {"papers"},
    },

    {
        "query": "Find research literature about multi-agent coordination.",
        "expected_agents": {"papers"},
    },

    # =====================================================
    # SINGLE AGENT — WEB
    # =====================================================

    {
        "query": "Find the official LangGraph documentation about persistence.",
        "expected_agents": {"web"},
    },

    {
        "query": "Explain how LangGraph checkpointing works.",
        "expected_agents": {"web"},
    },

    {
        "query": "What are the current features of Qdrant?",
        "expected_agents": {"web"},
    },

    # =====================================================
    # PAPERS + GITHUB
    # =====================================================

    {
        "query": (
            "Find papers and open-source implementations "
            "of multi-agent memory."
        ),
        "expected_agents": {
            "papers",
            "github",
        },
    },

    {
        "query": (
            "Find academic research on GraphRAG and "
            "GitHub implementations of it."
        ),
        "expected_agents": {
            "papers",
            "github",
        },
    },

    # =====================================================
    # WEB + PAPERS
    # =====================================================

    {
        "query": (
            "Research recent developments in AI agent memory "
            "and relevant academic papers."
        ),
        "expected_agents": {
            "web",
            "papers",
        },
    },

    {
        "query": (
            "Explain recent advances in RAG security and "
            "find relevant research papers."
        ),
        "expected_agents": {
            "web",
            "papers",
        },
    },

    # =====================================================
    # WEB + GITHUB
    # =====================================================

    {
        "query": (
            "Find open-source RAG frameworks and explain "
            "their current features."
        ),
        "expected_agents": {
            "github",
            "web",
        },
    },

    {
        "query": (
            "Explain the current LangGraph ecosystem and "
            "find relevant GitHub repositories."
        ),
        "expected_agents": {
            "web",
            "github",
        },
    },

    # =====================================================
    # WEB + PAPERS + GITHUB
    # =====================================================

    {
        "query": (
            "How does agent memory work? Explain the architecture "
            "and find relevant research papers and GitHub "
            "implementations."
        ),
        "expected_agents": {
            "web",
            "papers",
            "github",
        },
    },

    {
        "query": (
            "Research Retrieval-Augmented Generation, explain "
            "current approaches, find academic papers, and "
            "show open-source implementations."
        ),
        "expected_agents": {
            "web",
            "papers",
            "github",
        },
    },

    # =====================================================
    # MEMORY
    # =====================================================

    {
        "query": (
            "What did we previously find about "
            "multi-agent memory systems?"
        ),
        "expected_agents": {
            "memory",
        },
    },

    {
        "query": (
            "Recall our previous research about "
            "Retrieval-Augmented Generation."
        ),
        "expected_agents": {
            "memory",
        },
    },

    # =====================================================
    # MEMORY + EXTERNAL RESEARCH
    # =====================================================

    {
        "query": (
            "Compare what we previously found about agent memory "
            "with recent developments on the web."
        ),
        "expected_agents": {
            "memory",
            "web",
        },
    },
]