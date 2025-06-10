import json
import vector_db
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_transformers import LongContextReorder
import torch

def _truncate(text: str, max_length: int = 512) -> str:
    """Truncate text to a maximum length."""
    if len(text) > max_length:
        return text[:max_length]
    return text

class CrossEncoder:
    def __init__(self, vector_db: vector_db.VectorDB = None):
        self.db = vector_db
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.x_encoder = HuggingFaceCrossEncoder(model_name='BAAI/bge-reranker-base', model_kwargs={'device': self.device})

    def answer_query(self, query:str, answer_list:list, reordered_length: int = 10):
        pairs = [
            (query, _truncate(item['entity']['text']))
            for item in answer_list[0]
        ]

        scores = self.x_encoder.score(pairs)

        docs = [{"name": item['entity']['name'], "text": item['entity']['text']}
                for item in answer_list[0]]

        reranked_results = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )
        new_results = [result for result, _ in reranked_results]
        reordered = LongContextReorder().transform_documents(new_results)

        return reordered[:reordered_length], json.dumps(reordered[:reordered_length])


if __name__ == '__main__':
    cross_encoder = CrossEncoder()
    query = "What is the law about?"
    result = cross_encoder.answer_query(query)
    print(len(result))