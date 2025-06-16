from download import EurlexDownloader
from vector_db import VectorDB
from utils import DEFAULT_SAVE_FILE, DEFAULT_EURLEX_URL, DEFAULT_EVAL_FILE, load_json
import os
from embedding import EmbeddingModel
from evalutation import Evaluation
from create_dataset import flatten_and_select_docs
import time
from cross_encoder import CrossEncoder
from comparison import RetrievalComparison

if __name__ == "__main__":
    # 1. Download data from EUR-Lex and save it to a file
    start_time = time.time()
    if not os.path.exists(DEFAULT_SAVE_FILE):
        downloader = EurlexDownloader(DEFAULT_EURLEX_URL)
        data = downloader()
        downloader.save_to_json(data, DEFAULT_SAVE_FILE)
    end_time = time.time()
    print(f"Downloaded data in {end_time - start_time} seconds")

    # 2. Initialize bi-encoder
    embedding_model = EmbeddingModel()

    # 3. Create a vector database from the downloaded data
    start_time = time.time()
    vector_db = VectorDB(embedding_model=embedding_model)
    if not vector_db.collection_exists():
        vector_db.populate_db_from_json_file(DEFAULT_SAVE_FILE)
    end_time = time.time()
    print(f"VectorDB initialized in {end_time - start_time} seconds")

    # 4. Generate questions and answers
    if not os.path.exists(DEFAULT_EVAL_FILE):
        selected_docs = flatten_and_select_docs(data, selection_probability=0.01)
        eval_test = Evaluation(context_list=selected_docs)
        eval_test(save_to_file=True)

    # 5. Load eval dataset
    eval_dataset = load_json(DEFAULT_EVAL_FILE)

    # 6. Create a cross-encoder
    cross_encoder = CrossEncoder()

    # 7. Run Retrieval Comparison
    retrieval_comparison = RetrievalComparison(
        dataset=eval_dataset,
        cross_encoder=cross_encoder,
        vector_db=vector_db,
    )

    result = retrieval_comparison()
    print(result)
