import logging
import os
import random

from openai import AsyncOpenAI
from pymilvus import MilvusClient

from comparison import RAGComparison, RetrievalComparison, AdapterRetrievalComparison
from cross_encoder import CrossEncoder
from download import EurlexDownloader
from embedding import EmbeddingModel
from evaluation import EvaluationDatasetGenerator
from settings import Settings
from utils import (
    TEST_FILE,
    DEFAULT_RAG_COMPARISON_FILE,
    DEFAULT_RETRIEVAL_COMPARISON_FILE,
    DEFAULT_SAVE_FILE,
    DEFAULT_ADAPTER_RETRIEVAL_COMPARISON_FILE,
    load_json,
    save_json,
)
from vector_db import VectorDB
import time

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

ADAPTER_FILE_PATH = "./data/adapters/lin2_train/"

async def main():
    # 0. Load settings from environment variables
    settings = Settings()

    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

    data = load_json(DEFAULT_SAVE_FILE)
    # Load test dataset
    test_dataset = load_json(TEST_FILE)

    # Create a vector database from the downloaded data
    embedding_model = EmbeddingModel(settings.embedding_model)
    milvus_client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token)
    
    file = "lin2_ep_57.zip"
    vector_db = VectorDB(embedding_model=embedding_model, milvus_client=milvus_client, adapter_path = os.path.join(ADAPTER_FILE_PATH, file))
    if not vector_db.collection_exists():
        vector_db.create_collection_from_documents(documents=data, drop_existing=True)


    # Run Retrieval Comparison
    start_time = time.time()
    retrieval_comparison = AdapterRetrievalComparison(
        dataset=test_dataset,
        vector_db=vector_db,
    )

    result = retrieval_comparison()
    logger.info(f"Retrieval (adapter: {file}) comparison time: {time.time() - start_time} seconds, ACCURACY: {result['accuracy']['with_adapter']}")
    save_json(result, DEFAULT_ADAPTER_RETRIEVAL_COMPARISON_FILE)

    # # 7. Run RAG evaluation
    # start_time = time.time()
    # rag_evaluation = RAGComparison(
    #     openai_client=openai_client,
    #     dataset=eval_dataset,
    #     vector_db=vector_db,
    #     cross_encoder=cross_encoder,
    #     model_name=settings.llm_model,
    # )
    # result = await rag_evaluation()
    # logger.info(f"RAG evaluation time: {time.time() - start_time} seconds")
    # save_json(result, DEFAULT_RAG_COMPARISON_FILE)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
