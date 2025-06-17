import json
import logging

import pymilvus as pym

from embedding import EmbeddingModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDB:
    def __init__(
        self, embedding_model: EmbeddingModel, milvus_client: pym.MilvusClient
    ) -> None:
        self.milvus_client = milvus_client
        self.embedding_model = embedding_model
        self.collection_name = "laws"

    def get_response(self, prompt: str, search_width: int = 10) -> tuple[list, str]:
        try:
            vector_prompt = self.embedding_model.model.encode(prompt)

            query_vector = self.milvus_client.search(
                collection_name=self.collection_name,
                data=[vector_prompt],
                search_params={"metric_type": "COSINE"},
                output_fields=["text", "name"],
                limit=search_width,
            )

            return query_vector, json.dumps(query_vector)
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            raise

    def create_collection_from_documents(
        self, documents: list[list[dict[str, str]]], drop_existing: bool = False
    ):
        try:
            if drop_existing:
                logger.info("Dropping existing collection")
                self.milvus_client.drop_collection(self.collection_name)
            logger.info("Calculating embeddings")
            docs_with_embeddings = self.embedding_model(documents)
            vector_size = docs_with_embeddings[0][0]["vector"].shape[0]
            self.create_collection(vector_size)
            self.insert_vectors(docs_with_embeddings)
            logger.info("Database population completed")
        except Exception as e:
            logger.error(f"Error populating database: {e}")
            raise

    def create_collection(self, vector_size: int):
        if not self.collection_exists():
            logger.info("Creating collection")
            collection_schema = pym.CollectionSchema(
                fields=[
                    pym.FieldSchema(
                        name="id",
                        dtype=pym.DataType.INT64,
                        is_primary=True,
                        auto_id=False,
                    ),
                    pym.FieldSchema(
                        name="vector",
                        dtype=pym.DataType.FLOAT_VECTOR,
                        dim=vector_size,
                    ),
                    pym.FieldSchema(
                        name="text", dtype=pym.DataType.VARCHAR, max_length=int(1e4)
                    ),
                    pym.FieldSchema(
                        name="name", dtype=pym.DataType.VARCHAR, max_length=int(3e3)
                    ),
                ],
                description="laws",
            )
            index_params = self.milvus_client.prepare_index_params()
            index_params.add_index(field_name="id", index_type="AUTOINDEX")
            index_params.add_index(
                field_name="vector", index_type="AUTOINDEX", metric_type="COSINE"
            )
            self.milvus_client.create_collection(
                collection_name=self.collection_name,
                dimension=vector_size,
                schema=collection_schema,
                index_params=index_params,
            )
            logger.info(f"Collection {self.collection_name} created")
        else:
            logger.info("Collection already exists")

    def collection_exists(self) -> bool:
        return self.milvus_client.has_collection(self.collection_name)

    def insert_vectors(
        self, docs_with_embeddings: list[list[dict[str, str]]], batch_size: int = 500
    ):
        data = []
        id = 0
        for law in docs_with_embeddings:
            for section in law:
                data.append(
                    {
                        "id": id,
                        "vector": section["vector"],
                        "text": section["text"],
                        "name": section["name"],
                    }
                )
                id += 1

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            self.milvus_client.insert(
                collection_name=self.collection_name, data=batch, progress_bar=True
            )
            logger.info(f"Inserted batch {i // batch_size + 1}, size: {len(batch)}")
