import time
from os.path import exists
import json
from vector_db import VectorDB
from cross_encoder import CrossEncoder
from evalutation import Evaluation


class Comparison:
    def __init__(self, vector_db: VectorDB = None, cross_encoder: CrossEncoder = None):
        self.vector_db = vector_db if vector_db else VectorDB()
        self.cross_encoder = cross_encoder if cross_encoder else CrossEncoder(self.vector_db)

        if exists('../filtered_data.json'):
            with open('../filtered_data.json', 'r') as f:
                raw_data = json.load(f)
                evaluate = Evaluation(raw_data[:3])
            self.dataset = evaluate()
        else:
            raise FileNotFoundError("Missing dataset at ../filtered_data.json")


    def _compare_responses(self, query: str, top_k: int = 10):
        start_bi_encoder = time.time()
        vector_response, vector_form = self.vector_db.get_response(query)
        end_bi_encoder = time.time()

        start_cross_encoder = time.time()
        reranked_response, reranked_form = self.cross_encoder.answer_query(query)
        end_cross_encoder = time.time()

        return {
            "bi_encoder_response": vector_response,
            "bi_encoder_latency": end_bi_encoder - start_bi_encoder,
            "cross_encoder_response": reranked_response,
            "cross_encoder_latency": end_cross_encoder - start_cross_encoder,
        }

    def evaluate_dataset(self):
        for item in self.dataset:
            query = item['question']
            response = self._compare_responses(query)
            item.update(response)

            print(item)
            break


if __name__ == '__main__':
    comparison = Comparison()
    comparison.evaluate_dataset()

