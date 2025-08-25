import asyncio
import logging

from openai import AsyncOpenAI

from cross_encoder import CrossEncoder
from evaluation import call_llm
from prompts import RAG_RESPONSE_PROMPT
from vector_db import VectorDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LawAssistant:
    def __init__(
        self,
        vector_db: VectorDB,
        cross_encoder: CrossEncoder,
        openai_client: AsyncOpenAI,
        model_name: str = "gpt-4.1-nano-2025-04-14",
    ):
        self.openai_client = openai_client
        self.db = vector_db
        self.cross_encoder = cross_encoder
        self.model_name = model_name
        self.messages: list[dict[str, str]] = []

    async def generate_response(
        self,
        query: str,
        use_reranker: bool = True,
        *,
        top_k: int = 5,
        multiplier: int = 2,
    ):
        try:
            if use_reranker:
                response, formatted = self.db.get_response(query, search_width=top_k * multiplier)
                _, formatted = self.cross_encoder.rerank_format_documents(query, response, top_k)
            else:
                _, formatted = self.db.get_response(query, search_width=top_k)

            prompt = RAG_RESPONSE_PROMPT.format(context=formatted, question=query)
            response = await call_llm(self.openai_client, prompt, self.model_name)
            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while processing your query: {str(e)}"

    def generate_response_sync(
        self,
        query: str,
        use_reranker: bool = True,
        *,
        top_k: int = 5,
        multiplier: int = 2,
    ):
        return asyncio.run(self.generate_response(query, use_reranker, top_k=top_k, multiplier=multiplier))
