import numpy as np
import pymilvus as pym
from streamlit import metric

import embedding

class VectorDB:
    def __init__(self, link:str=""):
        print("Initializing VectorDB")
        self.client = pym.MilvusClient(uri='http://localhost:19530', token='root:Milvus')
        self.collection_name = 'ustawy'
        self.link = link

    def __call__(self):
        self._fetch_vector()
        self.client.drop_collection(self.collection_name)
        self._create_collection()
        self._insert_vectors()

    def _fetch_vector(self):
        print("Fetching vector")
        emb = embedding.EmbeddingModel(self.link)
        print(emb.vector_ustaw)
        emb.get_embedding()
        self.ustawy = emb.vector_ustaw
        self.vector_size = self.ustawy[0].shape[1]
        print("Vector fetched")

    def _create_collection(self):
        if not self.client.has_collection(self.collection_name):
            print("Creating collection")
            self.client.create_collection(collection_name=self.collection_name, dimension=self.vector_size)
            print(f"Collection {self.collection_name} created")
        else:
            print("Collection already exists")

    def _insert_vectors(self):
        data = []
        id = 0
        for ustawa in self.ustawy:
            for punkt in ustawa:
                data.append({'id': id, 'vector': punkt.tolist()})
                id += 1

        res = self.client.insert(collection_name=self.collection_name, data=data)
        print(f"Inserted {len(data)} vectors into collection {self.collection_name}, {res}")

    def get_response(self, prompt):
        print("Getting response")
        emb = embedding.EmbeddingModel()
        vector_prompt = emb.model.encode(prompt)['dense_vecs'].tolist()
        query_vector = self.client.search(collection_name=self.collection_name, data=[vector_prompt], search_params={'metric_type': 'COSINE'}, output_fields=['vector'])
        return query_vector[0][0]
        
if __name__ == '__main__':
    db = VectorDB("https://eur-lex.europa.eu/search.html?lang=pl&text=industry&qid=1742919459451&type=quick&DTS_SUBDOM=LEGISLATION&scope=EURLEX&FM_CODED=REG&page=1")
    resp = db.get_response("test")
    print(len(resp['entity']['vector']))
