import json

from langchain_community.document_transformers import LongContextReorder
from sentence_transformers import CrossEncoder as SentenceTransformersCrossEncoder

from utils import get_device, truncate


class CrossEncoder:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.cross_encoder = SentenceTransformersCrossEncoder(
            model_name, device=get_device()
        )
        self.max_length = self.cross_encoder.max_length

    def rerank_format_documents(
        self, query: str, answer_list: list, reordered_length: int = 10
    ) -> tuple[list[dict], str]:
        selected_docs = self.rerank_documents(query, answer_list, reordered_length)
        reordered = LongContextReorder().transform_documents(selected_docs)
        return selected_docs, json.dumps(reordered)

    def rerank_documents(
        self, query: str, answer_list: list, reordered_length: int = 10
    ) -> list[dict]:
        docs = [
            {"name": item["entity"]["name"], "text": item["entity"]["text"]}
            for item in answer_list[0]
        ]
        pairs = [(query, truncate(doc["text"], self.max_length)) for doc in docs]
        scores = self.cross_encoder.predict(pairs)
        docs_with_scores = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in docs_with_scores[:reordered_length]]
