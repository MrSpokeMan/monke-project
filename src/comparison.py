import asyncio
import logging
import time

from openai import AsyncOpenAI

from cross_encoder import CrossEncoder
from evaluation import call_llm
from law_assistant import LawAssistant
from prompts import EVALUATION_PROMPT
from vector_db import VectorDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGComparison:
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        vector_db: VectorDB,
        cross_encoder: CrossEncoder,
        dataset: list[dict[str, str]],
        model_name: str = "gpt-4.1-nano-2025-04-14",
    ):
        self.vector_db = vector_db
        self.assistant = LawAssistant(
            vector_db=self.vector_db,
            openai_client=openai_client,
            cross_encoder=cross_encoder,
            model_name=model_name,
        )
        self.dataset = dataset
        self.openai_client = openai_client
        self.model_name = model_name

    async def __call__(self, top_k: int = 10) -> dict:
        with_reranker_score, with_reranker_time = await self._evaluate_retrieval_method(
            use_reranker=True, top_k=top_k
        )
        (
            without_reranker_score,
            without_reranker_time,
        ) = await self._evaluate_retrieval_method(use_reranker=False, top_k=top_k)
        return {
            "score": {
                "with_reranker": with_reranker_score,
                "without_reranker": without_reranker_score,
            },
            "time": {
                "with_reranker": with_reranker_time,
                "without_reranker": without_reranker_time,
            },
        }

    async def _evaluate_retrieval_method(
        self, use_reranker: bool, top_k: int = 10
    ) -> tuple[float, float]:
        total_time = 0.0
        total_score = 0
        responses = []
        for item in self.dataset:
            start_time = time.time()
            response = await self.assistant.generate_response(
                item["question"], use_reranker=use_reranker, top_k=top_k
            )
            responses.append(response)
            total_time += time.time() - start_time

        scoring_tasks = [
            self._evaluate_with_llm_as_judge(item["question"], response, item["answer"])
            for item, response in zip(self.dataset, responses)
        ]

        scoring_results = await asyncio.gather(*scoring_tasks, return_exceptions=True)

        for item, result in zip(self.dataset, scoring_results):
            if isinstance(result, Exception):
                continue
            score, _ = result  # type: ignore[misc]
            total_score += score

        return total_score / len(self.dataset), total_time / len(self.dataset)

    async def _evaluate_with_llm_as_judge(
        self, instruction: str, response: str, reference_answer: str
    ) -> tuple[int, str]:
        eval_prompt = EVALUATION_PROMPT.format(
            instruction=instruction,
            response=response,
            reference_answer=reference_answer,
        )
        eval_prompt = await call_llm(self.openai_client, eval_prompt, self.model_name)
        try:
            feedback, score = [item.strip() for item in eval_prompt.split("[RESULT]")]
            score = int(score)
            return score, feedback
        except ValueError:
            return 0, "Error parsing evaluation result"


class RetrievalComparison:
    def __init__(
        self,
        vector_db: VectorDB,
        cross_encoder: CrossEncoder,
        dataset: list[dict[str, str]],
    ):
        self.vector_db = vector_db
        self.cross_encoder = cross_encoder
        self.dataset = dataset

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
            "top_k": top_k,
        }

    def _retrieve_documents(
        self, query: str, use_reranker: bool = False, top_k: int = 10
    ) -> list:
        try:
            if use_reranker:
                return self._retrieve_with_reranker(query, top_k)
            else:
                return self._retrieve_without_reranker(query, top_k)
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return []

    def _retrieve_with_reranker(
        self, query: str, top_k: int, multiplier: int = 5
    ) -> list:
        retrieved_docs, _ = self.vector_db.get_response(
            query, search_width=top_k * multiplier
        )
        return self.cross_encoder.rerank_documents(
            query, retrieved_docs, reordered_length=top_k
        )

    def _retrieve_without_reranker(self, query: str, top_k: int) -> list:
        retrieved_docs, _ = self.vector_db.get_response(query, search_width=top_k)
        return [
            {"name": doc["entity"]["name"], "text": doc["entity"]["text"]}
            for doc in retrieved_docs[0][:top_k]
        ]

    def _evaluate_retrieval_method(
        self, use_reranker: bool, top_k: int = 10
    ) -> tuple[int, float]:
        correct_count = 0
        total_time = 0.0
        for item in self.dataset:
            start_time = time.time()
            retrieved_docs = self._retrieve_documents(
                item["question"], use_reranker, top_k
            )
            # Check if the expected context is in the retrieved documents:
            if any(item["context"] in doc["text"] for doc in retrieved_docs):
                correct_count += 1
            total_time += time.time() - start_time
        return correct_count, total_time
