import random

from utils import load_json, parse_cli_args, save_json


def flatten_and_select_docs(
    data: list[list[dict[str, str]]],
    selection_probability: float = 1.0,
) -> list[dict[str, str]]:
    flattened = [item for sublist in data for item in sublist]
    return [
        item
        for item in flattened
        if random.random() < selection_probability
        if item.get("text")
    ]


if __name__ == "__main__":
    args = parse_cli_args()

    assert args.source == "json", "Only JSON source is supported"
    data = load_json(args.path_or_url)
    selected = flatten_and_select_docs(data, args.probability)

    if args.save:
        save_json(selected, args.save)

    total_items = len(data)
    print(f"Selected {len(selected)} entries out of {total_items}.")
