import os

import ollama
from langchain_community.llms.ollama import Ollama
from tqdm.auto import tqdm
from prompts import QA_GENERATION_PROMPT, QA_CRITIQUE_GROUNDEDNESS, QA_CRITIQUE_RELEVANCE, QA_CRITIQUE_STANDALONE
from google import genai
from dotenv import load_dotenv
<<<<<<< Updated upstream
from create_dataset import EurlexSelector
=======
import requests
import json
>>>>>>> Stashed changes

load_dotenv()

class Evaluation:
    def __init__(self, context_list: list[dict[str, str]]):
        self.client = ollama.Client()
        try:
            self.genai_client = genai.Client()
        except ValueError:
            print("Google GenAI Client not initialized. Please check your environment variables. (Add .env file with GOOGLE_API_KEY = <your_key>)")
            raise
        self.context_list = context_list if context_list else []

    def __call__(self):
        result = self._generate_questions()
        result = self._fill_dataset(result)
        # We return only questions and answers that were evaluated minimally in every categories above 3
        result = _remove_low_scores(result)
        return result

    def _call_llm(self, query: str):
        response = self.client.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": query
                }
            ]
        )
        return response["message"]["content"]

    def _generate_questions(self):
        result_list = []
        for context in tqdm(self.context_list, desc="Generating questions and answers", unit="q_and_a"):
            result_dict = {"name": context["name"], "context": context["text"]}
            query = QA_GENERATION_PROMPT.format(context=context["text"])
            response = self._call_llm(query)
            try:
                result_dict["question"] = response.split("Factoid question: ")[1].split("Answer: ")[0].strip()
                result_dict["answer"] = response.split("Answer: ")[1].strip()
            except IndexError:
                print(f"Error parsing response: {response}")
                continue
            result_list.append(result_dict)
        return result_list

    def _fill_dataset(self, outputs):
        for output in tqdm(outputs, desc="Evaluating questions", unit="eval_q"):
            evaluations = {
                "groundedness": self._call_llm(QA_CRITIQUE_GROUNDEDNESS.format(question=output["question"], context=output["context"])),
                "relevance": self._call_llm(QA_CRITIQUE_RELEVANCE.format(question=output["question"])),
                "standalone": self._call_llm(QA_CRITIQUE_STANDALONE.format(question=output["question"]))
            }
            try:
                for criterion, evaluation in evaluations.items():
                    score, eval = (
                        int(evaluation.split("Total rating: ")[-1].strip()),
                        evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1].strip(),
                    )
                    output.update({f"{criterion}_score": score, f"{criterion}_eval": eval})
            except Exception as e:
                print(f"Error processing evaluation: {e}")
        return outputs

def _remove_low_scores(outputs):
    filtered_outputs = []
    for output in outputs:
        if (output["groundedness_score"] >= 3 and
            output["relevance_score"] >= 3 and
            output["standalone_score"] >= 3):
            filtered_outputs.append(output)
    return filtered_outputs

if __name__ == '__main__':
<<<<<<< Updated upstream
    selector = EurlexSelector(data="filtered_data.json")
    context_list = [item["text"] for item in selector.original_data if "text" in item]

    eval_test = Evaluation(context_list = context_list)
    eval_test()
=======
    eval_test = Evaluation(context_list = [{"name": "Test Law", "text": "This is a test law text for evaluation purposes."}])
    print(eval_test())
>>>>>>> Stashed changes
