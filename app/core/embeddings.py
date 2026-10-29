from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"
model = TextEmbedding(model_name=MODEL_NAME)


def generate_embedding(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    return [vector.tolist() for vector in model.embed(texts)]
