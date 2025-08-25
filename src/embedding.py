from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils import get_device


class EmbeddingModel:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.model = SentenceTransformer(model_name, device=get_device())

    def __call__(
        self, documents: list[list[dict[str, str]]]
    ) -> list[list[dict[str, str]]]:
        vector_laws = []
        for document in tqdm(documents, unit="law", desc="Processing laws"):
            processed_points = []
            for point in document:
                text = point.get("text", "")
                emb = self.model.encode(text)
                processed_points.append(point | {"vector": emb})
            vector_laws.append(processed_points)
        return vector_laws
