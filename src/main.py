import os
import random

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pymilvus import MilvusClient

from comparison import RAGComparison
from cross_encoder import CrossEncoder
from download import EurlexDownloader
from embedding import EmbeddingModel
from evaluation import EvaluationDatasetGenerator
from utils import DEFAULT_EURLEX_URL, DEFAULT_EVAL_FILE, DEFAULT_SAVE_FILE, load_json
from vector_db import VectorDB


def flatten_and_select_docs(
    data: list[list[dict[str, str]]],
    selection_probability: float = 1.0,
) -> list[dict[str, str]]:
    flattened = [item for sublist in data for item in sublist]
    return [
        item
        for item in flattened
        if random.random() < selection_probability
        if item.get("text")
    ]


async def main():
    # 0. Load environment variables
    load_dotenv()
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 1. Download data from EUR-Lex and save it to a file
    if not os.path.exists(DEFAULT_SAVE_FILE):
        downloader = EurlexDownloader(DEFAULT_EURLEX_URL)
        data = await downloader()
        downloader.save_to_json(data, DEFAULT_SAVE_FILE)
    data = load_json(DEFAULT_SAVE_FILE)

    # 2. Create a vector database from the downloaded data
    embedding_model = EmbeddingModel(
        os.getenv("EMBEDDING_MODEL_NAME"),
    )
    milvus_client = MilvusClient(
        uri=os.getenv("MILVUS_URI"), token=os.getenv("MILVUS_TOKEN")
    )
    vector_db = VectorDB(embedding_model=embedding_model, milvus_client=milvus_client)
    if not vector_db.collection_exists():
        vector_db.create_collection_from_documents(documents=data, drop_existing=True)

    # 3. Generate questions and answers
    if not os.path.exists(DEFAULT_EVAL_FILE):
        selected_docs = flatten_and_select_docs(data, selection_probability=0.005)
        eval_test = EvaluationDatasetGenerator(
            openai_client=openai_client, context_list=selected_docs
        )
        await eval_test(save_to_file=True)

    # 4. Load eval dataset
    eval_dataset = load_json(DEFAULT_EVAL_FILE)

    # 5. Create a cross-encoder
    cross_encoder = CrossEncoder(
        os.getenv("CROSS_ENCODER_MODEL_NAME"),
    )

    # 6. Run Retrieval Comparison
    # retrieval_comparison = RetrievalComparison(
    #     dataset=eval_dataset,
    #     cross_encoder=cross_encoder,
    #     vector_db=vector_db,
    # )

    # result = retrieval_comparison()
    # print(result)

    # 7. Run RAG evaluation
    rag_evaluation = RAGComparison(
        openai_client=openai_client,
        dataset=eval_dataset,
        vector_db=vector_db,
    )
    result = await rag_evaluation()
    print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
