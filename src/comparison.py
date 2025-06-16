import json
import time
from os.path import exists

from tqdm import tqdm

from create_dataset import EurlexSelector
from cross_encoder import CrossEncoder
from evalutation import Evaluation
from law_assistant import LawAssistant
from prompts import EVALUATION_PROMPT
from vector_db import VectorDB


class Comparison:
    def __init__(self, vector_db: VectorDB | None = None):
        self.vector_db = vector_db if vector_db else VectorDB()
        self.assistant = LawAssistant(vector_db=self.vector_db)

        if exists("./data/evaluation_results.json"):
            self.evaluate = Evaluation(openai_client=self.assistant.openai_client)
            with open("./data/evaluation_results.json", "r", encoding="utf-8") as f:
                self.dataset = json.load(f)
            print("Loaded dataset from ../data/evaluation_results.json")
        elif exists("./data/scraped_data.json"):
            selector = EurlexSelector(data="../data/scraped_data.json")
            context_list = [
                item[0] for item in selector.original_data if item[0].get("text")
            ]
            self.evaluate = Evaluation(context_list[:5], self.assistant.openai_client)
            self.dataset = self.evaluate()
        else:
            raise FileNotFoundError(
                "Missing dataset at ../data/scraped_data.json or ../data/evaluation_results.json"
            )

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

    def evaluate_dataset(self):
        print("Evaluating dataset...")

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
    def __init__(self, vector_db: VectorDB | None = None):
        self.vector_db = vector_db if vector_db else VectorDB()
        self.x_encoder = CrossEncoder(self.vector_db)

        if exists("./data/evaluation_results.json"):
            with open("./data/evaluation_results.json", "r", encoding="utf-8") as f:
                self.dataset = json.load(f)
            print("Loaded dataset from ./data/evaluation_results.json")
        else:
            raise FileNotFoundError("Missing dataset at ./data/evaluation_results.json")

    def _retrieve_documents(
        self, query: str, use_reranker: bool = False, top_k: int = 10
    ):
        try:
            if use_reranker:
                retrieved_docs, _ = self.vector_db.get_response(
                    query, search_width=top_k * 5
                )
                print(
                    "With Reranker, Fetched: ",
                    len(retrieved_docs[0]),
                    " Top K: ",
                    top_k,
                )
                result = self.x_encoder.rerank_documents(
                    query, retrieved_docs, reordered_length=top_k
                )
            else:
                retrieved_docs, _ = self.vector_db.get_response(
                    query, search_width=top_k
                )
                print(
                    "Without Reranker, Fetched: ",
                    len(retrieved_docs[0]),
                    " Top K: ",
                    top_k,
                )
                result = [
                    {"name": doc["entity"]["name"], "text": doc["entity"]["text"]}
                    for doc in retrieved_docs[0][:top_k]
                ]
            return result
        except Exception as e:
            print(f"Error in retrieval: {e}")
            return []

    def _check_retrieval_accuracy(
        self,
        query: str,
        expected_context: str,
        use_reranker: bool = False,
        top_k: int = 5,
    ):
        retrieved_docs = self._retrieve_documents(query, use_reranker, top_k)

        for doc in retrieved_docs:
            doc_text = doc["text"]
            if expected_context in doc_text:
                return True
        return False

    def __call__(self):
        import time

        correct_without_reranker = 0
        correct_with_reranker = 0
        total_time_without_reranker = 0
        total_time_with_reranker = 0

        for item in tqdm(self.dataset):
            query = item["question"]
            expected = item["context"]
            top_k = 10

            start_time = time.time()
            if self._check_retrieval_accuracy(
                query, expected, use_reranker=False, top_k=top_k
            ):
                correct_without_reranker += 1
            total_time_without_reranker += time.time() - start_time

            start_time = time.time()
            if self._check_retrieval_accuracy(
                query, expected, use_reranker=True, top_k=top_k
            ):
                correct_with_reranker += 1
            total_time_with_reranker += time.time() - start_time

        total_items = len(self.dataset)
        accuracy_without_reranker = correct_without_reranker / total_items
        accuracy_with_reranker = correct_with_reranker / total_items
        avg_time_without_reranker = total_time_without_reranker / total_items
        avg_time_with_reranker = total_time_with_reranker / total_items

        print("Retrieval Accuracy without reranker:", accuracy_without_reranker)
        print("Retrieval Accuracy with reranker:", accuracy_with_reranker)
        print("Average time without reranker:", avg_time_without_reranker)
        print("Average time with reranker:", avg_time_with_reranker)

        return {
            "accuracy_without_reranker": accuracy_without_reranker,
            "accuracy_with_reranker": accuracy_with_reranker,
            "avg_time_without_reranker": avg_time_without_reranker,
            "avg_time_with_reranker": avg_time_with_reranker,
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
    retrieval_comparison()
