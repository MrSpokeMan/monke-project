import pymilvus as pym
import json
import embedding

class VectorDB:
    def __init__(self, link:str=""):
        print("Initializing VectorDB")
        self.client = pym.MilvusClient(uri='http://localhost:19530', token='root:Milvus')
        self.collection_name = 'laws'
        self.link = link

    def __call__(self):
        self._fetch_vector()
        self.client.drop_collection(self.collection_name) # Uncomment to drop the collection, if needed
        self._create_collection()
        self._insert_vectors()

    def _fetch_vector(self):
        print("Fetching vector")
        if self.link:
            emb = embedding.EmbeddingModel(self.link)
            emb.get_embedding()
            self.ustawy = emb.vector_ustaw
            self.vector_size = self.ustawy[0][0]['vector'].shape[0]
            print("Vector size: ", self.vector_size)
            print("Vector fetched")
        else:
            print("No link, skipping vector fetching")

    def _create_collection(self):
        if not self.client.has_collection(self.collection_name):
            print("Creating collection")
            # define collection schema
            collection_schema = pym.CollectionSchema(
                fields=[
                    pym.FieldSchema(name='id', dtype=pym.DataType.INT64, is_primary=True, auto_id=False),
                    pym.FieldSchema(name='vector', dtype=pym.DataType.FLOAT_VECTOR, dim=self.vector_size),
                    pym.FieldSchema(name='text', dtype=pym.DataType.VARCHAR, max_length=int(1e4)),
                    pym.FieldSchema(name="name", dtype=pym.DataType.VARCHAR, max_length=int(1e3))
                ],
                description="ustawy"
            )
            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name='id', index_type='AUTOINDEX')
            index_params.add_index(field_name='vector', index_type='AUTOINDEX', metric_type='COSINE')
            self.client.create_collection(collection_name=self.collection_name, dimension=self.vector_size, schema=collection_schema, index_params=index_params)
            print(f"Collection {self.collection_name} created")
        else:
            print("Collection already exists")

    def _insert_vectors(self):
        data = []
        id = 0
        for ustawa in self.ustawy:
            for punkt in ustawa:
                data.append({'id': id,
                             'vector': punkt['vector'].tolist(),
                             'text': punkt['text'],
                             'name': punkt['name']
                             })
                id += 1

        res = self.client.insert(collection_name=self.collection_name, data=data, progress_bar=True)
        print(f"Inserted {len(data)} vectors into collection {self.collection_name}, {res}")

    def get_response(self, prompt):
        # Getting response from the bot
        print("Getting response")
        emb = embedding.EmbeddingModel()
        vector_prompt = emb.model.encode(prompt)['dense_vecs'].tolist()

        query_vector = self.client.search(collection_name=self.collection_name,
                                          data=[vector_prompt],
                                          search_params={'metric_type': 'COSINE'},
                                          output_fields=['text', 'name'],
                                          limit=5
                                          )

        return query_vector, json.dumps(query_vector)
        
if __name__ == '__main__':
    db = VectorDB("https://eur-lex.europa.eu/search.html?lang=en&text=industry&qid=1742919459451&type=quick&DTS_SUBDOM=LEGISLATION&scope=EURLEX&FM_CODED=REG")
    db()
    # print(len(resp['entity']['vector']))
