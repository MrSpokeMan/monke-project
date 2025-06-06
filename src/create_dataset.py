import download
import json
import random
import argparse
import os
import sys
from typing import List, Dict, Optional, Union
from cli_utils import parse_cli_args, DEFAULT_EURLEX_URL, DEFAULT_SAVE_FILE


class EurlexSelector:
    def __init__(
        self,
        data: Optional[Union[List[List[Dict[str, str]]], str]] = None,
        selection_probability: float = 1.0,
        seed: Optional[int] = None
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
            item for item in self.original_data
            if random.random() < self.selection_probability
        ]

    def save_to_json(self, filepath: str):
        """Saves the selected data to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.selected_data, f, ensure_ascii=False, indent=2)

    def load_from_json(self, filepath: str):
        """Loads data from a JSON file and sets it as original_data."""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.original_data = json.load(f)

    def __call__(self, filepath: Optional[str] = None):
        """Executes filtering and optionally saves to JSON."""
        if self.original_data and isinstance(self.original_data[0], list):
            self._flatten_and_store(self.original_data)
        self.filter_data()
        if filepath:
            self.save_to_json(filepath)
        return self.selected_data


if __name__ == "__main__":

    # Temporarily strip custom args before calling parse_cli_args
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]] + [
        arg for arg in sys.argv[1:]
        if not arg.startswith("--seed") and not arg.startswith("--probability")
    ]
    args = parse_cli_args()
    sys.argv = original_argv

    # Parse extra args (seed and probability)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--probability", type=float, default=0.05)
    extra_args, _ = parser.parse_known_args()

    seed = extra_args.seed if extra_args.seed is not None else random.randint(0, 2 ** 32 - 1)
    probability = extra_args.probability

    selector = EurlexSelector(
        data=args.path_or_url,
        selection_probability=probability,
        seed=seed
    )

    if args.save:
        save_name = args.save
        if args.save == "scraped_data.json":
            save_name = f"test_dataset_{seed}_{probability}"

        save_path = os.path.join(os.path.dirname(__file__), "..", "test", save_name)
        save_path = os.path.abspath(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        selector(filepath=save_path)
    else:
        selector()

    print(f"Selected {len(selector.selected_data)} entries out of {len(selector.original_data)}.")