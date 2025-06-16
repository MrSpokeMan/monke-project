import json


def validate_fields(json_path, text_limit=10000, name_limit=3000):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data and isinstance(data[0], list):
        data = [item for sublist in data for item in sublist]

    for i, item in enumerate(data):
        name = item.get("name", "")
        text = item.get("text", "")

        if len(text.encode("utf-8")) > text_limit:
            print(f"[{i}] ❗ Text too long ({len(text.encode('utf-8'))} bytes)")

        if len(name.encode("utf-8")) > name_limit:
            print(
                f"[{i}] ⚠️ Name too long ({len(name.encode('utf-8'))} bytes): {name[:80]}..."
            )


if __name__ == "__main__":
    validate_fields("scraped_data.json")
