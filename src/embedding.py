from FlagEmbedding import BGEM3FlagModel
import torch
from tqdm import tqdm

import download as download

class EmbeddingModel():
    def __init__(self, url:str=""):
        if url != "":
            self.down = download.EurlexDownloader(url)
            self.ustawy = self.down()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(device)
        self.model = BGEM3FlagModel('BAAI/bge-m3', devices=device)
        self.vector_ustaw = []

    def get_embedding(self):
        for ustawa in tqdm(self.ustawy, unit="law", desc="Processing laws"):
            points = []
            #for point in tqdm(ustawa, unit="point", desc="Processing points in law"):
            for point in ustawa:
                text = point.get("text", "")
                emb = self.model.encode(text)
                point['vector'] = emb['dense_vecs']
                points.append(point)
            self.vector_ustaw.append(points)

if __name__ == '__main__':

    print("CUDA available:", torch.cuda.is_available())
    print("Torch version:", torch.__version__)
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
    emb = EmbeddingModel("https://eur-lex.europa.eu/search.html?lang=pl&text=industry&qid=1742919459451&type=quick&DTS_SUBDOM=LEGISLATION&scope=EURLEX&FM_CODED=REG")
    emb.get_embedding()
    print("vector: ", emb.vector_ustaw, "rozmiar: ", emb.vector_ustaw.shape)
