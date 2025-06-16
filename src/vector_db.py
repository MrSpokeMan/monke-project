import json

import pymilvus as pym

import embedding
from src.utils import parse_cli_args


class VectorDB:
    def __init__(self, source="web", link="", json_path="", save_path=""):
        print("Initializing VectorDB")
        self.client = pym.MilvusClient(
            uri="http://localhost:19530", token="root:Milvus"
        )
        self.collection_name = "laws"
        self.source = source
        self.link = link
        self.json_path = json_path
        self.save_path = save_path

    def __call__(self):
        self._fetch_vector()
        self.client.drop_collection(self.collection_name)
        self._create_collection()
        self._insert_vectors()

    def _fetch_vector(self):
        print("Fetching vector")
        emb = embedding.EmbeddingModel(
            source=self.source,
            path_or_url=(self.json_path if self.source == "json" else self.link),
            save_json_path=self.save_path,
            download_data=True,
        )
        emb.get_embedding()
        self.laws = emb.vector_laws
        self.vector_size = self.laws[0][0]["vector"].shape[0]
        print("Vector size:", self.vector_size)
        print("Vector fetched")

    def _create_collection(self):
        if not self.client.has_collection(self.collection_name):
            print("Creating collection")
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
                        dim=self.vector_size,
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
            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name="id", index_type="AUTOINDEX")
            index_params.add_index(
                field_name="vector", index_type="AUTOINDEX", metric_type="COSINE"
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.vector_size,
                schema=collection_schema,
                index_params=index_params,
            )
            print(f"Collection {self.collection_name} created")
        else:
            print("Collection already exists")

    def _insert_vectors(self, batch_size: int = 500):
        data = []
        id = 0
        for law in self.laws:
            for section in law:
                data.append(
                    {
                        "id": id,
                        "vector": section["vector"].tolist(),
                        "text": section["text"],
                        "name": section["name"],
                    }
                )
                id += 1

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            self.client.insert(
                collection_name=self.collection_name, data=batch, progress_bar=True
            )
            print(f"Inserted batch {i // batch_size + 1}, size: {len(batch)}")

    def get_response(self, prompt, search_width=10):
        emb = embedding.EmbeddingModel()
        vector_prompt = emb.model.encode(prompt)

        query_vector = self.client.search(
            collection_name=self.collection_name,
            data=[vector_prompt],
            search_params={"metric_type": "COSINE"},
            output_fields=["text", "name"],
            limit=search_width,
        )

        return query_vector, json.dumps(query_vector)


if __name__ == "__main__":
    args = parse_cli_args()
    db = VectorDB(
        source=args.source,
        link=args.path_or_url if args.source == "web" else "",
        json_path=args.path_or_url if args.source == "json" else "",
        save_path=args.save or "",
    )
    db()
