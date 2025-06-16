from typing import Literal

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import download as download
from cli_utils import DEFAULT_EURLEX_URL, parse_cli_args
from utils import get_device


class EmbeddingModel:
    def __init__(
        self,
        source: Literal["web", "json"] = "web",
        path_or_url: str = DEFAULT_EURLEX_URL,
        save_json_path: str = "",
        download_data: bool = False,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.vector_documents: list[list[dict[str, str]]] = []
        self.device = get_device()
        self.model = SentenceTransformer(model_name, device=self.device)

        if download_data:
            if source == "web":
                self.downloader = download.EurlexDownloader(path_or_url)
                self.documents: list[list[dict[str, str]]] = self.downloader()
                if save_json_path:
                    self.downloader.save_to_json(self.documents, save_json_path)

            elif source == "json":
                self.downloader = download.EurlexDownloader("")
                self.documents = self.downloader.load_from_json(path_or_url)

            else:
                raise ValueError("Invalid source. Choose 'web' or 'json'.")

    def get_embedding(self):
        for document in tqdm(self.documents, unit="law", desc="Processing laws"):
            points = []
            for point in document:
                text = point.get("text", "")
                emb = self.model.encode(text)
                point["vector"] = emb
                points.append(point)
            self.vector_documents.append(points)


if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    print("Torch version:", torch.__version__)
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

    args = parse_cli_args()

    model = EmbeddingModel(
        source=args.source, path_or_url=args.path_or_url, save_json_path=args.save or ""
    )
    model.get_embedding()
