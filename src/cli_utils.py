import argparse

DEFAULT_EURLEX_URL = (
    "https://eur-lex.europa.eu/search.html?lang=pl&text=industry&qid=1742919459451&type=quick&DTS_SUBDOM=LEGISLATION&scope=EURLEX&FM_CODED=REG"
)

DEFAULT_SAVE_FILE = "scraped_data.json"


def parse_cli_args():
    parser = argparse.ArgumentParser(description="EUR-Lex Legal Document Processor")

    parser.add_argument(
        "source",
        choices=["web", "json"],
        help="Choose 'web' to scrape from EUR-Lex or 'json' to load from a saved file."
    )

    parser.add_argument(
        "path_or_url",
        nargs="?",
        help="If 'web', optional URL (defaults to EUR-Lex). If 'json', optional path to JSON file (defaults to saved file)."
    )

    parser.add_argument(
        "--save",
        nargs="?",
        const=DEFAULT_SAVE_FILE,
        help=f"If source is 'web', save scraped data (default: {DEFAULT_SAVE_FILE})"
    )

    args = parser.parse_args()

    # Assign defaults if path_or_url is missing
    if args.source == "web" and not args.path_or_url:
        args.path_or_url = DEFAULT_EURLEX_URL
    elif args.source == "json" and not args.path_or_url:
        args.path_or_url = DEFAULT_SAVE_FILE

    return args