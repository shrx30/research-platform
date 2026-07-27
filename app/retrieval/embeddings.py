from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)


def embed(text: str):

    return embedding_model.encode(text).tolist()