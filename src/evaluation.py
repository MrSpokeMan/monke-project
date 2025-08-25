import asyncio
import json
import logging

from openai import AsyncOpenAI
from tqdm.auto import tqdm

from prompts import (
    QA_CRITIQUE_GROUNDEDNESS,
    QA_CRITIQUE_RELEVANCE,
    QA_CRITIQUE_STANDALONE,
    QA_GENERATION_PROMPT,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def call_llm(openai_client: AsyncOpenAI, query: str, model_name: str):
    completion = await openai_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": query}],
    )
    return completion.choices[0].message.content


class EvaluationDatasetGenerator:
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        model_name: str,
        context_list: list[dict[str, str]],
    ):
        self.openai_client = openai_client
        self.context_list = context_list
        self.model_name = model_name

    async def __call__(self, file_path: str | None = None):
        result = await self._generate_questions()
        result = await self._fill_dataset(result)
        result = _remove_low_scores(result)
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
        return result

    async def _generate_single_question(self, context: dict[str, str]):
        result_dict = {"name": context["name"], "context": context["text"]}
        query = QA_GENERATION_PROMPT.format(context=context["text"])
        try:
            response = await call_llm(self.openai_client, query, self.model_name)
            result_dict["question"] = response.split("Factoid question: ")[1].split("Answer: ")[0].strip()
            result_dict["answer"] = response.split("Answer: ")[1].strip()
            return result_dict
        except (IndexError, AttributeError) as e:
            logger.error(f"Error parsing response for context '{context['name']}': {e}")
            return None

    async def _generate_questions(self):
        tasks = [self._generate_single_question(context) for context in self.context_list]
        results = []
        with tqdm(total=len(tasks), desc="Generating Q&A", unit="q_and_a") as pbar:
            batch_size = 10
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i : i + batch_size]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Error in batch processing: {result}")
                    elif result is not None:
                        results.append(result)
                    pbar.update(1)

        return results

    async def _evaluate_single_output(self, output: dict):
        evaluation_tasks = {
            "groundedness": call_llm(
                self.openai_client,
                QA_CRITIQUE_GROUNDEDNESS.format(question=output["question"], context=output["context"]),
                self.model_name,
            ),
            "relevance": call_llm(
                self.openai_client,
                QA_CRITIQUE_RELEVANCE.format(question=output["question"]),
                self.model_name,
            ),
            "standalone": call_llm(
                self.openai_client,
                QA_CRITIQUE_STANDALONE.format(question=output["question"]),
                self.model_name,
            ),
        }

        try:
            evaluations = await asyncio.gather(*[task for task in evaluation_tasks.values()], return_exceptions=True)
            for criterion, evaluation in zip(evaluation_tasks.keys(), evaluations):
                if isinstance(evaluation, Exception):
                    logger.error(
                        f"Error evaluating {criterion} for question '{output['question'][:50]}...': {evaluation}"
                    )
                    continue

                score, eval_text = (
                    int(evaluation.split("Total rating: ")[-1].strip()),  # type: ignore[union-attr]
                    evaluation.split("Total rating: ")[-2]  # type: ignore[union-attr]
                    .split("Evaluation: ")[1]
                    .strip(),
                )
                output.update({f"{criterion}_score": score, f"{criterion}_eval": eval_text})
        except Exception as e:
            logger.error(f"Error processing evaluation for question '{output['question'][:50]}...': {e}")

        return output

    async def _fill_dataset(self, outputs):
        logger.info("Evaluating questions...")
        tasks = [self._evaluate_single_output(output) for output in outputs]
        results = []
        with tqdm(total=len(tasks), desc="Evaluating questions", unit="eval_q") as pbar:
            batch_size = 5
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i : i + batch_size]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Error in evaluation batch: {result}")
                    else:
                        results.append(result)
                    pbar.update(1)
        return results


def _remove_low_scores(
    outputs: list[dict],
    min_groundedness_score: int = 3,
    min_relevance_score: int = 3,
    min_standalone_score: int = 3,
):
    filtered_outputs = []
    for output in outputs:
        if output.get("groundedness_score") and output.get("relevance_score") and output.get("standalone_score"):
            if (
                output["groundedness_score"] >= min_groundedness_score
                and output["relevance_score"] >= min_relevance_score
                and output["standalone_score"] >= min_standalone_score
            ):
                filtered_outputs.append(output)
    return filtered_outputs
