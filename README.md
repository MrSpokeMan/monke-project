# Advanced RAG with Rerankers

This repository demonstrates the application of rerankers as an Advanced RAG (Retrieval-Augmented Generation) technique to improve retrieval performance. The project showcases how reranking can enhance the quality of retrieved documents before they are passed to the generation model.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast and reliable dependency management.

### Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Install uv
```bash
pip install uv
```

### Install Dependencies
```bash
uv sync
```

This will create a virtual environment and install all required dependencies automatically.

## Setup

### 1. Start Milvus Vector Database
Before running the application, you need to start the Milvus vector database:

```bash
cd milvus
docker-compose up -d
```

This will start Milvus using Docker Compose with the configuration provided in `milvus/docker-compose.yml`.

### 2. Set up Environment Variables
Copy the example environment file and configure your API key:

```bash
cp .env.example .env
```

Then edit the `.env` file and replace `your_openai_api_key_here` with your actual OpenAI API key. You can obtain an API key from the [OpenAI Platform](https://platform.openai.com/api-keys).

## Usage

To run the complete pipeline, execute:

```bash
uv run src/main.py
```

This will run the complete RAG pipeline with reranking evaluation:

1. **Data Download**: Downloads EUR-Lex legal documents
2. **Vector Database Setup**: Creates embeddings using sentence transformers and stores them in Milvus
3. **Evaluation Dataset Generation**: Generates question-answer pairs from selected documents using OpenAI's LLM
4. **Cross-Encoder Reranking**: Initializes a BAAI/bge-reranker model for improving retrieval results
5. **Retrieval Comparison**: Tests retrieval performance with and without reranking across different top-k values (3, 5, 10, 15)
6. **RAG Evaluation**: Performs end-to-end RAG evaluation comparing baseline retrieval vs. reranked retrieval for answer generation

The results are saved as JSON files in the `data/` directory for analysis.
