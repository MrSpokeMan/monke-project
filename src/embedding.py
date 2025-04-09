from FlagEmbedding import BGEM3FlagModel
import download as download
import numpy as np

class EmbeddingModel():
    def __init__(self, url):
        self.down = download.EurlexDownloader(url)
        self.ustawy = self.down()
        self.model = BGEM3FlagModel('BAAI/bge-m3')
        self.vector_ustaw = np.empty(len(self.ustawy), dtype=object)
        
    def get_embedding(self):
        for idx, ustawa in enumerate(self.ustawy):
            emb = self.model.encode(ustawa)
            self.vector_ustaw[idx] = emb['dense_vecs']

if __name__ == '__main__':
    emb = EmbeddingModel("https://eur-lex.europa.eu/search.html?lang=pl&text=industry&qid=1742919459451&type=quick&DTS_SUBDOM=LEGISLATION&scope=EURLEX&FM_CODED=REG&page=1")
    emb.get_embedding()
    print("vector: ", emb.vector_ustaw, "rozmiar: ", emb.vector_ustaw.shape)
