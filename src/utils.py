import json
from os.path import exists
from typing import Literal
import torch
from torch import nn
from sentence_transformers import SentenceTransformer

DEFAULT_EURLEX_URL = "https://eur-lex.europa.eu/search.html?lang=en&text=industry&qid=1742919459451&type=quick&DTS_SUBDOM=LEGISLATION&scope=EURLEX&FM_CODED=REG"

DEFAULT_SAVE_FILE = "./data/scraped_data.json"

DEFAULT_EVAL_FILE = "./data/evaluation_results.json"

DEFAULT_RETRIEVAL_COMPARISON_FILE = "./data/retrieval_comparison.json"

DEFAULT_ADAPTER_RETRIEVAL_COMPARISON_FILE = "./data/adapter_retrieval_comparison.json"

DEFAULT_RAG_COMPARISON_FILE = "./data/rag_comparison.json"

TRAIN_FILE = "./data/TRAIN.json"
TEST_FILE = "./data/TEST.json"


class LinearAdapter(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.linear(x)

class TwoLayerAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim  # Możesz ustawić np. na input_dim // 2

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

def load_adapter(adapter_path:str, base_model: SentenceTransformer) -> SentenceTransformer:
    loaded_dict = torch.load(adapter_path)
    loaded_adapter = TwoLayerAdapter(base_model.get_sentence_embedding_dimension())
    device = get_device()
    loaded_adapter.load_state_dict(loaded_dict['adapter_state_dict'])
    return loaded_adapter.to(device)


def get_device() -> Literal["cuda", "mps", "cpu"]:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def truncate(text: str, max_length: int = 512) -> str:
    if len(text) > max_length:
        return text[:max_length]
    return text


def load_json(path: str) -> list | dict:
    if not exists(path):
        raise FileNotFoundError(f"Missing file at {path}")

    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset


def save_json(data: list | dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
