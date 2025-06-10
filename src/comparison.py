import time
from os.path import exists
from prompts import EVALUATION_PROMPT
from vector_db import VectorDB
from law_assistant import LawAssistant
from evalutation import Evaluation
from create_dataset import EurlexSelector


class Comparison:
    def __init__(self, vector_db: VectorDB = None):
        self.vector_db = vector_db if vector_db else VectorDB()
        self.assistant = LawAssistant(vector_db=self.vector_db)

        if exists('../data/scraped_data.json'):
            selector = EurlexSelector(data="../data/scraped_data.json")
            context_list = [item[0] for item in selector.original_data if item[0].get("text")]
            self.evaluate = Evaluation(context_list[:50])
            self.dataset = self.evaluate()
        else:
            raise FileNotFoundError("Missing dataset at ../data/scraped_data.json")


    def _compare_responses(self, query: str):
        start_bi_encoder = time.time()
        bot_bi, bi_return = self.assistant.generate_response(query, reranker=False, full_resturn=True)
        end_bi_encoder = time.time()

        start_cross_encoder = time.time()
        bot_cross, cross_return = self.assistant.generate_response(query, reranker=True, full_resturn=True)
        end_cross_encoder = time.time()

        return {
            "bi_encoder_assistant_response": bot_bi,
            "bi_retrieved_documents": bi_return[0],
            "bi_encoder_latency": round(end_bi_encoder - start_bi_encoder, 4),
            "cross_encoder_assistant_response": bot_cross,
            "cross_retrieved_documents": cross_return,
            "cross_encoder_latency": round(end_cross_encoder - start_cross_encoder, 4)
        }

    def evaluate_dataset(self):
        all_bi_answers = []
        all_cross_answers = []
        results = []

        print("Evaluating dataset...")

        for item in self.dataset:
            query = item['question']

            expected = item['answer']

            response = self._compare_responses(query)
            item.update(response)

            for evaluator in ["bi", "cross"]:
                eval_prompt = EVALUATION_PROMPT.format(
                    instruction=query,
                    response=response[f"{evaluator}_encoder_assistant_response"],
                    reference_answer=expected
                )
                eval_prompt = self.evaluate.call_llm(eval_prompt)
                try:
                    feedback, score = [item.strip() for item in eval_prompt.split("[RESULT]")]
                except ValueError:
                    feedback = eval_prompt
                    score = 0

                try:
                    score = int(score)
                except ValueError:
                    score = 0

                item[f"{evaluator}_encoder_feedback"] = feedback
                item[f"{evaluator}_encoder_score"] = score

            bi_results = [doc['entity'] for doc in item['bi_retrieved_documents']]
            cross_results = [doc for doc in item['cross_retrieved_documents']]

            all_bi_answers.append((expected, bi_results))
            all_cross_answers.append((expected, cross_results))

            results.append(item)

        bi_metrics = _calculate_metrics(all_bi_answers)
        cross_metrics = _calculate_metrics(all_cross_answers)

        return {
            "results": results,
            "metrics": {
                "bi_encoder": bi_metrics,
                "cross_encoder": cross_metrics
            }
        }

def calculate_accuracy(results):
    bi_accuracy = 0
    cross_accuracy = 0
    bi_avg_latency = 0
    cross_avg_latency = 0
    for item in results["results"]:
        bi_accuracy += item['bi_encoder_score']
        cross_accuracy += item['cross_encoder_score']
        bi_avg_latency += item['bi_encoder_latency']
        cross_avg_latency += item['cross_encoder_latency']

    print("Bi-Encoder Accuracy:", bi_accuracy / len(results))
    print("Cross-Encoder Accuracy:", cross_accuracy / len(results))
    print("Bi-Encoder Average Latency:", bi_avg_latency / len(results))
    print("Cross-Encoder Average Latency:", cross_avg_latency / len(results))

def _calculate_metrics(answers):
    total = len(answers)
    correct = sum(1 for expected, response in answers if expected in response)
    mrr = sum(1 / (response.index(expected) + 1) for expected, response in answers if expected in response) / total if total > 0 else 0

    precision = correct / total if total > 0 else 0
    recall = correct / total if total > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "total": total,
        "correct": correct,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mrr": mrr
        }


if __name__ == '__main__':
    comparison = Comparison()
    res = comparison.evaluate_dataset()
    calculate_accuracy(res)

