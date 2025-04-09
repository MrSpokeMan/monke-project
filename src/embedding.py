from FlagEmbedding import BGEM3FlagModel
import download as download
import numpy as np
import time

class EmbeddingModel():
    def __init__(self, url:str=""):
        if url != "":
            self.down = download.EurlexDownloader(url)
            self.ustawy = self.down()
        self.model = BGEM3FlagModel('BAAI/bge-m3', return_sparse=True)
        self.vector_ustaw = []

    def get_embedding(self):
        for ustawa in self.ustawy:
            for point in ustawa:
                text = point.get("text", "")
                emb = self.model.encode(text)
                self.vector_ustaw.append(emb['dense_vecs'])

if __name__ == '__main__':
    emb = EmbeddingModel("https://eur-lex.europa.eu/search.html?lang=pl&text=industry&qid=1742919459451&type=quick&DTS_SUBDOM=LEGISLATION&scope=EURLEX&FM_CODED=REG")
    emb.get_embedding()
    vecs = np.array(emb.vector_ustaw)
    print("vector: ", vecs, "rozmiar: ", vecs.shape)
