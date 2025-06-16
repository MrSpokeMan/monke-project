import torch
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_transformers import LongContextReorder

import vector_db


def _truncate(text: str, max_length: int = 512) -> str:
    """Truncate text to a maximum length."""
    if len(text) > max_length:
        return text[:max_length]
    return text


class CrossEncoder:
    def __init__(self, vector_db: vector_db.VectorDB | None = None):
        self.db = vector_db
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.x_encoder = HuggingFaceCrossEncoder(
            model_name="BAAI/bge-reranker-v2-m3", model_kwargs={"device": self.device}
        )

    def answer_query(self, query: str, answer_list: list, reordered_length: int = 10):
        new_results = self.rerank_documents(query, answer_list, reordered_length)
        reordered = LongContextReorder().transform_documents(new_results)
        return reordered

    def rerank_documents(
        self, query: str, answer_list: list, reordered_length: int = 10
    ):
        pairs = [(query, _truncate(item["entity"]["text"])) for item in answer_list[0]]

        scores = self.x_encoder.score(pairs)

        docs = self._milvus_response_to_docs(answer_list)

        reranked_results = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [result for result, _ in reranked_results[:reordered_length]]

    def _milvus_response_to_docs(self, response: list) -> list[dict[str, str]]:
        return [
            {"name": item["entity"]["name"], "text": item["entity"]["text"]}
            for item in response[0]
        ]
