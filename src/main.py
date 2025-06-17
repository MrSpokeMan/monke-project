import os
import random
import time

from comparison import RAGComparison
from cross_encoder import CrossEncoder
from download import EurlexDownloader
from embedding import EmbeddingModel
from evalutation import Evaluation
from utils import DEFAULT_EURLEX_URL, DEFAULT_EVAL_FILE, DEFAULT_SAVE_FILE, load_json
from vector_db import VectorDB
from dotenv import load_dotenv
from openai import AsyncOpenAI


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
    start_time = time.time()
    if not os.path.exists(DEFAULT_SAVE_FILE):
        downloader = EurlexDownloader(DEFAULT_EURLEX_URL)
        data = await downloader()
        downloader.save_to_json(data, DEFAULT_SAVE_FILE)
    else:
        data = load_json(DEFAULT_SAVE_FILE)
    end_time = time.time()
    print(f"Downloaded data in {end_time - start_time} seconds")

    # 2. Create a vector database from the downloaded data
    start_time = time.time()
    embedding_model = EmbeddingModel()
    vector_db = VectorDB(embedding_model=embedding_model)
    if not vector_db.collection_exists():
        vector_db.populate_db_from_json_file(DEFAULT_SAVE_FILE)
    end_time = time.time()
    print(f"VectorDB initialized in {end_time - start_time} seconds")

    # 3. Generate questions and answers
    if not os.path.exists(DEFAULT_EVAL_FILE):
        selected_docs = flatten_and_select_docs(data, selection_probability=0.01)
        eval_test = Evaluation(openai_client=openai_client, context_list=selected_docs)
        await eval_test(save_to_file=True)

    # 4. Load eval dataset
    eval_dataset = load_json(DEFAULT_EVAL_FILE)

    # 5. Create a cross-encoder
    cross_encoder = CrossEncoder()

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
