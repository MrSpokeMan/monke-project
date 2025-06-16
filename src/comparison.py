import time

from tqdm import tqdm

from cross_encoder import CrossEncoder
from law_assistant import LawAssistant
from prompts import EVALUATION_PROMPT
from utils import load_json
from vector_db import VectorDB


class RAGComparison:
    def __init__(
        self, vector_db: VectorDB, dataset_path: str = "./data/evaluation_results.json"
    ):
        self.vector_db = vector_db
        self.assistant = LawAssistant(vector_db=self.vector_db)
        self.dataset = load_json(dataset_path)

    def _compare_responses(self, query: str):
        start_bi_encoder = time.time()
        bot_bi = self.assistant.generate_response_research(query, use_reranker=False)
        end_bi_encoder = time.time()

        start_cross_encoder = time.time()
        bot_cross = self.assistant.generate_response_research(query, use_reranker=True)
        end_cross_encoder = time.time()

        return {
            "bi_encoder_assistant_response": bot_bi,
            "bi_encoder_latency": round(end_bi_encoder - start_bi_encoder, 4),
            "cross_encoder_assistant_response": bot_cross,
            "cross_encoder_latency": round(end_cross_encoder - start_cross_encoder, 4),
        }

    def _evaluate_rag(self, query: str, expected: str) -> dict:
        response = self._compare_responses(query)
        return {
            "bi_encoder_assistant_response": response["bi_encoder_assistant_response"],
            "bi_encoder_latency": response["bi_encoder_latency"],
            "cross_encoder_assistant_response": response[
                "cross_encoder_assistant_response"
            ],
        }

    def evaluate_dataset(self):
        for item in tqdm(self.dataset):
            query = item["question"]
            expected = item["answer"]

            response = self._compare_responses(query)
            item.update(response)

            for evaluator in ["bi", "cross"]:
                eval_prompt = EVALUATION_PROMPT.format(
                    instruction=query,
                    response=response[f"{evaluator}_encoder_assistant_response"],
                    reference_answer=expected,
                )
                eval_prompt = self.evaluate.call_llm(eval_prompt)
                try:
                    feedback, score = [
                        item.strip() for item in eval_prompt.split("[RESULT]")
                    ]
                except ValueError:
                    feedback = eval_prompt
                    score = 0

                try:
                    score = int(score)
                except ValueError:
                    score = 0

                item[f"{evaluator}_encoder_feedback"] = feedback
                item[f"{evaluator}_encoder_score"] = score

    def calculate_accuracy(self):
        bi_accuracy = 0
        cross_accuracy = 0
        bi_avg_latency = 0
        cross_avg_latency = 0
        for item in self.dataset:
            bi_accuracy += item["bi_encoder_score"]
            cross_accuracy += item["cross_encoder_score"]
            bi_avg_latency += item["bi_encoder_latency"]
            cross_avg_latency += item["cross_encoder_latency"]

        print("Bi-Encoder Accuracy:", bi_accuracy / len(self.dataset))
        print("Cross-Encoder Accuracy:", cross_accuracy / len(self.dataset))
        print("Bi-Encoder Average Latency:", bi_avg_latency / len(self.dataset))
        print("Cross-Encoder Average Latency:", cross_avg_latency / len(self.dataset))


class RetrievalComparison:
    def __init__(
        self, vector_db: VectorDB, dataset_path: str = "./data/evaluation_results.json"
    ):
        self.vector_db = vector_db
        self.x_encoder = CrossEncoder()
        self.dataset = load_json(dataset_path)

    def _retrieve_documents(
        self, query: str, use_reranker: bool = False, top_k: int = 10
    ) -> list:
        try:
            if use_reranker:
                return self._retrieve_with_reranker(query, top_k)
            else:
                return self._retrieve_without_reranker(query, top_k)
        except Exception as e:
            print(f"Error in retrieval: {e}")
            return []

    def _retrieve_with_reranker(
        self, query: str, top_k: int, multiplier: int = 5
    ) -> list:
        retrieved_docs, _ = self.vector_db.get_response(
            query, search_width=top_k * multiplier
        )
        return self.x_encoder.rerank_documents(
            query, retrieved_docs, reordered_length=top_k
        )

    def _retrieve_without_reranker(self, query: str, top_k: int) -> list:
        retrieved_docs, _ = self.vector_db.get_response(query, search_width=top_k)
        return [
            {"name": doc["entity"]["name"], "text": doc["entity"]["text"]}
            for doc in retrieved_docs[0][:top_k]
        ]

    def _check_retrieval_accuracy(
        self,
        query: str,
        expected_context: str,
        use_reranker: bool = False,
        top_k: int = 5,
    ) -> bool:
        retrieved_docs = self._retrieve_documents(query, use_reranker, top_k)
        return any(expected_context in doc["text"] for doc in retrieved_docs)

    def _evaluate_retrieval_method(
        self, use_reranker: bool, top_k: int = 10
    ) -> tuple[int, float]:
        correct_count = 0
        total_time = 0.0
        for item in self.dataset:
            start_time = time.time()
            if self._check_retrieval_accuracy(
                query=item["question"],
                expected_context=item["context"],
                use_reranker=use_reranker,
                top_k=top_k,
            ):
                correct_count += 1
            total_time += time.time() - start_time
        return correct_count, total_time

    def __call__(self, top_k: int = 10) -> dict:
        total_items = len(self.dataset)
        correct_without, time_without = self._evaluate_retrieval_method(
            use_reranker=False, top_k=top_k
        )
        correct_with, time_with = self._evaluate_retrieval_method(
            use_reranker=True, top_k=top_k
        )
        return {
            "accuracy": {
                "without_reranker": correct_without / total_items,
                "with_reranker": correct_with / total_items,
            },
            "avg_time": {
                "without_reranker": time_without / total_items,
                "with_reranker": time_with / total_items,
            },
        }


if __name__ == "__main__":
    print("Processing data...")
    vector_db = VectorDB(source="json", json_path="./data/scraped_data.json")
    # vector_db()
    print("VectorDB initialized")

    # comparison = Comparison(vector_db)
    # comparison.evaluate_dataset()
    # comparison.calculate_accuracy()

    retrieval_comparison = RetrievalComparison(vector_db)
    result = retrieval_comparison()
    print(result)
