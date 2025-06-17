import asyncio
import logging

from openai import AsyncOpenAI

from cross_encoder import CrossEncoder
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
    ):
        self.openai_client = openai_client
        self.db = vector_db
        self.cross_encoder = cross_encoder
        self.messages: list[dict[str, str]] = []

    async def generate_response(self, query: str, use_reranker: bool = True):
        try:
            if use_reranker:
                response, formatted = self.db.get_response(query, search_width=50)
                _, formatted = self.cross_encoder.rerank_format_documents(
                    query, response
                )
            else:
                _, formatted = self.db.get_response(query)

            prompt = RAG_RESPONSE_PROMPT.format(context=formatted, question=query)
            response = await self.openai_client.chat.completions.create(
                model="gpt-4.1-nano-2025-04-14",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while processing your query: {str(e)}"

    def generate_response_sync(self, query: str, use_reranker: bool = True):
        return asyncio.run(self.generate_response(query, use_reranker))
