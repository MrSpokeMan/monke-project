import time
import json
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

        if exists('../data/evaluation_results.json'):
            self.evaluate = Evaluation(openai_client=self.assistant.openai_client)
            with open('../data/evaluation_results.json', 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)
            print("Loaded dataset from ../data/evaluation_results.json")
        elif exists('../data/scraped_data.json'):
            selector = EurlexSelector(data="../data/scraped_data.json")
            context_list = [item[0] for item in selector.original_data if item[0].get("text")]
            self.evaluate = Evaluation(context_list[:5], self.assistant.openai_client)
            self.dataset = self.evaluate()
        else:
            raise FileNotFoundError("Missing dataset at ../data/scraped_data.json or ../data/evaluation_results.json")


    def _compare_responses(self, query: str):
        start_bi_encoder = time.time()
        bot_bi = self.assistant.generate_response_research(query, reranker=False)
        end_bi_encoder = time.time()

        start_cross_encoder = time.time()
        bot_cross = self.assistant.generate_response_research(query, reranker=True)
        end_cross_encoder = time.time()

        return {
            "bi_encoder_assistant_response": bot_bi,
            "bi_encoder_latency": round(end_bi_encoder - start_bi_encoder, 4),
            "cross_encoder_assistant_response": bot_cross,
            "cross_encoder_latency": round(end_cross_encoder - start_cross_encoder, 4)
        }

    def evaluate_dataset(self):
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

    def calculate_accuracy(self):
        bi_accuracy = 0
        cross_accuracy = 0
        bi_avg_latency = 0
        cross_avg_latency = 0
        for item in self.dataset:
            bi_accuracy += item['bi_encoder_score']
            cross_accuracy += item['cross_encoder_score']
            bi_avg_latency += item['bi_encoder_latency']
            cross_avg_latency += item['cross_encoder_latency']

        print("Bi-Encoder Accuracy:", bi_accuracy / len(self.dataset))
        print("Cross-Encoder Accuracy:", cross_accuracy / len(self.dataset))
        print("Bi-Encoder Average Latency:", bi_avg_latency / len(self.dataset))
        print("Cross-Encoder Average Latency:", cross_avg_latency / len(self.dataset))

if __name__ == '__main__':
    comparison = Comparison()
    comparison.evaluate_dataset()
    comparison.calculate_accuracy()

