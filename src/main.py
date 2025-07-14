import logging
import os
import random

from openai import AsyncOpenAI
from pymilvus import MilvusClient

from comparison import RAGComparison, RetrievalComparison
from cross_encoder import CrossEncoder
from download import EurlexDownloader
from embedding import EmbeddingModel
from evaluation import EvaluationDatasetGenerator
from settings import Settings
from utils import (
    DEFAULT_EURLEX_URL,
    DEFAULT_EVAL_FILE,
    DEFAULT_RAG_COMPARISON_FILE,
    DEFAULT_RETRIEVAL_COMPARISON_FILE,
    DEFAULT_SAVE_FILE,
    load_json,
    save_json,
)
from vector_db import VectorDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def flatten_and_select_docs(
    data: list[list[dict[str, str]]],
    selection_probability: float = 1.0,
    seed: int = 42,
) -> list[dict[str, str]]:
    random.seed(seed)
    flattened = [item for sublist in data for item in sublist]
    return [
        item
        for item in flattened
        if random.random() < selection_probability
        if item.get("text")
    ]


async def main():
    # 0. Load settings from environment variables
    settings = Settings()

    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

    # 1. Download data from EUR-Lex and save it to a file
    if not os.path.exists(DEFAULT_SAVE_FILE):
        downloader = EurlexDownloader(DEFAULT_EURLEX_URL)
        data = await downloader()
        downloader.save_to_json(data, DEFAULT_SAVE_FILE)
    data = load_json(DEFAULT_SAVE_FILE)

    # 2. Create a vector database from the downloaded data
    embedding_model = EmbeddingModel(settings.embedding_model)
    milvus_client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token)
    vector_db = VectorDB(embedding_model=embedding_model, milvus_client=milvus_client)
    if not vector_db.collection_exists():
        vector_db.create_collection_from_documents(documents=data, drop_existing=True)

    # 3. Generate questions and answers
    if not os.path.exists(DEFAULT_EVAL_FILE):
        selected_docs = flatten_and_select_docs(data, selection_probability=1)#0.005)
        eval_test = EvaluationDatasetGenerator(
            openai_client=openai_client,
            context_list=selected_docs,
            model_name=settings.llm_model,
        )
        await eval_test(file_path=DEFAULT_EVAL_FILE)

    # 4. Load eval dataset
    eval_dataset = load_json(DEFAULT_EVAL_FILE)

    # 5. Create a cross-encoder
    cross_encoder = CrossEncoder(
        settings.cross_encoder_model,
    )

    # 6. Run Retrieval Comparison
    import time

    start_time = time.time()
    retrieval_comparison = RetrievalComparison(
        dataset=eval_dataset,
        cross_encoder=cross_encoder,
        vector_db=vector_db,
    )

    result = retrieval_comparison()
    logger.info(f"Retrieval comparison time: {time.time() - start_time} seconds")
    save_json(result, DEFAULT_RETRIEVAL_COMPARISON_FILE)

    # 7. Run RAG evaluation
    start_time = time.time()
    rag_evaluation = RAGComparison(
        openai_client=openai_client,
        dataset=eval_dataset,
        vector_db=vector_db,
        cross_encoder=cross_encoder,
        model_name=settings.llm_model,
    )
    result = await rag_evaluation()
    logger.info(f"RAG evaluation time: {time.time() - start_time} seconds")
    save_json(result, DEFAULT_RAG_COMPARISON_FILE)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
