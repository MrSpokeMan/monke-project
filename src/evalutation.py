import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.auto import tqdm

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
        openai_client: AsyncOpenAI | None = None,
    ):
        load_dotenv()
        try:
            self.openai_client = (
                openai_client
                if openai_client
                else AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            )
        except ValueError:
            print(
                "OpenAI Client not initialized. Please check your environment variables. (Add .env file with OPENAI_API_KEY = <your_key>)"
            )
            raise
        self.context_list = context_list if context_list else []

    async def __call__(self, save_to_file: bool = False):
        result = await self._generate_questions()
        result = await self._fill_dataset(result)
        result = _remove_low_scores(result)
        if save_to_file:
            with open("./data/evaluation_results.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            print("Results saved to evaluation_results.json")
        return result

    async def call_llm(self, query: str):
        completion = await self.openai_client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[{"role": "user", "content": query}],
        )
        return completion.choices[0].message.content

    async def _generate_single_question(self, context: dict[str, str]):
        """Generate a single question-answer pair for a given context."""
        result_dict = {"name": context["name"], "context": context["text"]}
        query = QA_GENERATION_PROMPT.format(context=context["text"])
        try:
            response = await self.call_llm(query)
            result_dict["question"] = (
                response.split("Factoid question: ")[1].split("Answer: ")[0].strip()
            )
            result_dict["answer"] = response.split("Answer: ")[1].strip()
            return result_dict
        except (IndexError, AttributeError) as e:
            print(f"Error parsing response for context '{context['name']}': {e}")
            return None

    async def _generate_questions(self):
        print("Generating questions and answers...")
        tasks = [
            self._generate_single_question(context) for context in self.context_list
        ]
        results = []
        with tqdm(total=len(tasks), desc="Generating Q&A", unit="q_and_a") as pbar:
            batch_size = 10
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i : i + batch_size]
                batch_results = await asyncio.gather(
                    *batch_tasks, return_exceptions=True
                )

                for result in batch_results:
                    if isinstance(result, Exception):
                        print(f"Error in batch processing: {result}")
                    elif result is not None:
                        results.append(result)
                    pbar.update(1)

        return results

    async def _evaluate_single_output(self, output: dict):
        evaluation_tasks = {
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
            evaluations = await asyncio.gather(
                *[task for task in evaluation_tasks.values()], return_exceptions=True
            )
            evaluation_results = dict(zip(evaluation_tasks.keys(), evaluations))
            for criterion, evaluation in evaluation_results.items():
                if isinstance(evaluation, Exception):
                    print(
                        f"Error evaluating {criterion} for question '{output['question'][:50]}...': {evaluation}"
                    )
                    continue

                score, eval_text = (
                    int(evaluation.split("Total rating: ")[-1].strip()),
                    evaluation.split("Total rating: ")[-2]
                    .split("Evaluation: ")[1]
                    .strip(),
                )
                output.update(
                    {f"{criterion}_score": score, f"{criterion}_eval": eval_text}
                )
        except Exception as e:
            print(
                f"Error processing evaluation for question '{output['question'][:50]}...': {e}"
            )

        return output

    async def _fill_dataset(self, outputs):
        print("Evaluating questions...")
        tasks = [self._evaluate_single_output(output) for output in outputs]
        results = []
        with tqdm(total=len(tasks), desc="Evaluating questions", unit="eval_q") as pbar:
            batch_size = 5
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i : i + batch_size]
                batch_results = await asyncio.gather(
                    *batch_tasks, return_exceptions=True
                )
                for result in batch_results:
                    if isinstance(result, Exception):
                        print(f"Error in evaluation batch: {result}")
                    else:
                        results.append(result)
                    pbar.update(1)
        return results


def _remove_low_scores(outputs):
    filtered_outputs = []
    for output in outputs:
        if (
            output.get("groundedness_score")
            and output.get("relevance_score")
            and output.get("standalone_score")
        ):
            if (
                output["groundedness_score"] >= 3
                and output["relevance_score"] >= 3
                and output["standalone_score"] >= 3
            ):
                filtered_outputs.append(output)
    return filtered_outputs
