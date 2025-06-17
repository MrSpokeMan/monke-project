import json
from os.path import exists
from typing import Literal

import torch

DEFAULT_EURLEX_URL = "https://eur-lex.europa.eu/search.html?lang=en&text=industry&qid=1742919459451&type=quick&DTS_SUBDOM=LEGISLATION&scope=EURLEX&FM_CODED=REG"

DEFAULT_SAVE_FILE = "./data/scraped_data.json"

DEFAULT_EVAL_FILE = "./data/evaluation_results.json"


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
