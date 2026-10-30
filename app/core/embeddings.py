from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"
model = TextEmbedding(model_name=MODEL_NAME)


def generate_embedding(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    return [vector.tolist() for vector in model.embed(texts)]


def generate_query_embedding(query: str):
    if not query.strip():
        raise ValueError("Query must not be empty")

    return generate_embedding([query])[0]
