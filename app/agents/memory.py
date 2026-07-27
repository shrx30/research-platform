from app.Tools.memory_tools import search_memory


def run(query: str) -> str:

    memories = search_memory(
        query=query,
        limit=5,
        min_score=0.65,
    )

    if not memories:
        return "No memory context."

    sections = []

    for memory in memories:
        sections.append(
            f"""
Topic: {memory["topic"]}
Content: {memory["content"]}
Confidence: {memory["confidence"]}
Sources: {", ".join(memory["sources"])}
Similarity: {memory["score"]:.3f}
""".strip()
        )

    return "\n\n---\n\n".join(sections)