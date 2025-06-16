import json
import random
from typing import Dict, List, Optional, Union

import download
from src.utils import parse_cli_args


class EurlexSelector:
    def __init__(
        self,
        data: Optional[Union[List[List[Dict[str, str]]], str]] = None,
        selection_probability: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.selection_probability = selection_probability
        self.seed = seed
        self.selected_data: List[Dict[str, str]] = []
        self.original_data: List[Dict[str, str]] = []

        if seed is not None:
            random.seed(seed)

        if isinstance(data, str):
            self.load_from_json(data)
        elif isinstance(data, list):
            self._flatten_and_store(data)

    def _flatten_and_store(self, data: List[List[Dict[str, str]]]):
        """Flattens list of lists and stores as original data."""
        self.original_data = [item for sublist in data for item in sublist]

    def filter_data(self):
        """Filters original_data based on the selection probability."""
        self.selected_data = [
            item
            for item in self.original_data
            if random.random() < self.selection_probability
        ]

    def save_to_json(self, filepath: str):
        """Saves the selected data to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.selected_data, f, ensure_ascii=False, indent=2)

    def load_from_json(self, filepath: str):
        """Loads data from a JSON file and sets it as original_data."""
        with open(filepath, "r", encoding="utf-8") as f:
            self.original_data = json.load(f)

    def __call__(self, filepath: Optional[str] = None):
        """Executes filtering and optionally saves to JSON."""
        self.filter_data()
        if filepath:
            self.save_to_json(filepath)
        return self.selected_data


if __name__ == "__main__":
    args = parse_cli_args()

    if args.source == "web":
        downloader = download.EurlexDownloader(args.path_or_url)
        data = downloader()
    else:
        downloader = download.EurlexDownloader("")
        data = downloader.load_from_json(args.path_or_url)

    selector = EurlexSelector(data, selection_probability=args.probability)
    selected = selector(args.save)
    print(f"Selected {len(selected)} entries out of {sum(len(x) for x in data)}.")
