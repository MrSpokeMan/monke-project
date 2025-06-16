import json
import os

import ollama
from dotenv import load_dotenv
from openai import OpenAI
from tqdm.auto import tqdm

from create_dataset import EurlexSelector
from prompts import (
    QA_CRITIQUE_GROUNDEDNESS,
    QA_CRITIQUE_RELEVANCE,
    QA_CRITIQUE_STANDALONE,
    QA_GENERATION_PROMPT,
)


class Evaluation:
    def __init__(
        self,
        context_list: list[dict[str, str]] | None = None,
        openai_client: OpenAI | None = None,
    ):
        load_dotenv()
        self.client = ollama.Client()
        try:
            self.openai_client = (
                openai_client
                if openai_client
                else OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            )
        except ValueError:
            print(
                "OpenAI Client not initialized. Please check your environment variables. (Add .env file with OPENAI_API_KEY = <your_key>)"
            )
            raise
        self.context_list = context_list if context_list else []

    def __call__(self, save_to_file: bool = False):
        result = self._generate_questions()
        result = self._fill_dataset(result)
        # We return only questions and answers that were evaluated minimally in every categories above 3
        result = _remove_low_scores(result)
        if save_to_file:
            with open("../data/evaluation_results.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            print("Results saved to evaluation_results.json")
        return result

    def call_llm(self, query: str):
        completion = self.openai_client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[{"role": "user", "content": query}],
        )
        return completion.choices[0].message.content
        # response = self.client.chat(
        #     model="llama3.2",
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": query
        #         }
        #     ]
        # )
        # return response["message"]["content"]

    def _generate_questions(self):
        result_list = []
        for context in tqdm(
            self.context_list, desc="Generating questions and answers", unit="q_and_a"
        ):
            result_dict = {"name": context["name"], "context": context["text"]}
            query = QA_GENERATION_PROMPT.format(context=context["text"])
            response = self.call_llm(query)
            try:
                result_dict["question"] = (
                    response.split("Factoid question: ")[1].split("Answer: ")[0].strip()
                )
                result_dict["answer"] = response.split("Answer: ")[1].strip()
            except IndexError:
                print(f"Error parsing response: {response}")
                continue
            result_list.append(result_dict)
        return result_list

    def _fill_dataset(self, outputs):
        for output in tqdm(outputs, desc="Evaluating questions", unit="eval_q"):
            evaluations = {
                "groundedness": self.call_llm(
                    QA_CRITIQUE_GROUNDEDNESS.format(
                        question=output["question"], context=output["context"]
                    )
                ),
                "relevance": self.call_llm(
                    QA_CRITIQUE_RELEVANCE.format(question=output["question"])
                ),
                "standalone": self.call_llm(
                    QA_CRITIQUE_STANDALONE.format(question=output["question"])
                ),
            }
            try:
                for criterion, evaluation in evaluations.items():
                    score, eval = (
                        int(evaluation.split("Total rating: ")[-1].strip()),
                        evaluation.split("Total rating: ")[-2]
                        .split("Evaluation: ")[1]
                        .strip(),
                    )
                    output.update(
                        {f"{criterion}_score": score, f"{criterion}_eval": eval}
                    )
            except Exception as e:
                print(f"Error processing evaluation: {e}")
        return outputs


def _remove_low_scores(outputs):
    filtered_outputs = []
    for output in outputs:
        if (
            output["groundedness_score"]
            and output["relevance_score"]
            and output["standalone_score"]
        ):
            if (
                output["groundedness_score"] >= 3
                and output["relevance_score"] >= 3
                and output["standalone_score"] >= 3
            ):
                filtered_outputs.append(output)
    return filtered_outputs


if __name__ == "__main__":
    selector = EurlexSelector(data="../data/scraped_data.json")
    context_list = [item for item in selector.original_data if item.get("text")]

    eval_test = Evaluation(context_list=context_list[:100])
    eval_test(save_to_file=True)
