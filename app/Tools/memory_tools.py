from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from app.config.settings import settings


COLLECTION_NAME = "research_memory"

# 384-dimensional embeddings
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)


def ensure_collection() -> None:
    """Create the research memory collection if it doesn't exist."""

    if client.collection_exists(COLLECTION_NAME):
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
    )


def store_memory(
    content: str,
    topic: str,
    sources: list[str] | None = None,
    confidence: float = 1.0,
) -> str:
    ensure_collection()

    memory_id = str(uuid4())

    vector = embedding_model.encode(
        content,
        normalize_embeddings=True,
    ).tolist()

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=memory_id,
                vector=vector,
                payload={
                    "type": "finding",
                    "topic": topic,
                    "content": content,
                    "sources": sources or [],
                    "confidence": confidence,
                },
            )
        ],
    )

    return memory_id

def search_memory(
    query: str,
    limit: int = 3,
    min_score: float = 0.65,
) -> list[dict]:
    """Retrieve semantically relevant research memories."""

    ensure_collection()

    vector = embedding_model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    points = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=limit,
        score_threshold=min_score,
    ).points

    memories = []

    for point in points:
        if not point.payload:
            continue

        memories.append(
            {
                "id": str(point.id),
                "topic": point.payload.get("topic", ""),
                "content": point.payload.get("content", ""),
                "sources": point.payload.get("sources", []),
                "confidence": point.payload.get("confidence", 1.0),
                "score": point.score,
            }
        )

    return memories