from FlagEmbedding import BGEM3FlagModel
import torch
from tqdm import tqdm
from cli_utils import DEFAULT_EURLEX_URL, parse_cli_args

import download as download

class EmbeddingModel:
    def __init__(self, source="web", path_or_url=DEFAULT_EURLEX_URL, save_json_path="", download_data:bool = False):
        self.vector_ustaw = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BGEM3FlagModel('BAAI/bge-m3', devices=self.device)

        if download_data:
            if source == "web":
                self.down = download.EurlexDownloader(path_or_url)
                self.ustawy = self.down()
                if save_json_path:
                    self.down.save_to_json(self.ustawy, save_json_path)

            elif source == "json":
                self.down = download.EurlexDownloader("")
                self.ustawy = self.down.load_from_json(path_or_url)

            else:
                raise ValueError("Invalid source. Choose 'web' or 'json'.")

    def get_embedding(self):
        for ustawa in tqdm(self.ustawy, unit="law", desc="Processing laws"):
            points = []
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

    args = parse_cli_args()

    model = EmbeddingModel(
        source=args.source,
        path_or_url=args.path_or_url,
        save_json_path=args.save or ""
    )
    model.get_embedding()
