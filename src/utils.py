import torch
from typing import Literal
import json
from os.path import exists


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
