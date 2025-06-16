import argparse
import json
from os.path import exists
from typing import Literal

import torch

DEFAULT_EURLEX_URL = "https://eur-lex.europa.eu/search.html?lang=en&text=industry&qid=1742919459451&type=quick&DTS_SUBDOM=LEGISLATION&scope=EURLEX&FM_CODED=REG"

DEFAULT_SAVE_FILE = "./data/scraped_data.json"


def parse_cli_args():
    parser = argparse.ArgumentParser(description="EUR-Lex Legal Document Processor")

    parser.add_argument(
        "source",
        choices=["web", "json"],
        default="json",
        help="Choose 'web' to scrape from EUR-Lex or 'json' to load from a saved file.",
    )

    parser.add_argument(
        "path_or_url",
        nargs="?",
        default=DEFAULT_SAVE_FILE,
        help="If 'web', optional URL (defaults to EUR-Lex). If 'json', optional path to JSON file (defaults to saved file).",
    )

    parser.add_argument(
        "--save",
        nargs="?",
        const=DEFAULT_SAVE_FILE,
        help=f"If source is 'web', save scraped data (default: {DEFAULT_SAVE_FILE})",
    )

    parser.add_argument(
        "--probability",
        type=float,
        default=0.05,
        help="Selection probability (default: 0.05)",
    )

    args = parser.parse_args()

    if args.source == "web" and not args.path_or_url:
        args.path_or_url = DEFAULT_EURLEX_URL
    elif args.source == "json" and not args.path_or_url:
        args.path_or_url = DEFAULT_SAVE_FILE

    return args


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
