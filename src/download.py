import json
import os
import re

import requests
from bs4 import BeautifulSoup

from cli_utils import DEFAULT_EURLEX_URL, DEFAULT_SAVE_FILE, parse_cli_args
from template_parser import (
    parse_template_1_first_format,
    parse_template_2_second_format,
    parse_template_3_third_format,
    parse_template_4_fourth_format,
)


class EurlexDownloader:
    def __init__(self, search_url: str):
        self.search_url = search_url
        self.all_documents: list[list[dict[str, str]]] = []

    def __call__(self):
        self.download_eurlex_page()
        return self.all_documents

    def save_to_json(self, data, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved scraped data to {path}")

    def load_from_json(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No such JSON file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded data from {path}")
        return data

    def get_last_page_number(self) -> int:
        response = requests.get(self.search_url)
        soup = BeautifulSoup(response.text, "html.parser")
        last_page_link = soup.find("a", title="Last Page")

        if last_page_link:
            href = last_page_link.get("href", "")
            match = re.search(r"page=(\d+)", href)
            if match:
                return int(match.group(1))
        return 1

    def download_eurlex_page(self) -> None:
        end_page = self.get_last_page_number()
        start_page = 0
        # end_page = 100
        print(end_page - start_page)
        for page in range(start_page, end_page):
            response = requests.get(self.search_url + f"&page={page + 1}")
            print(f"Fetching: {self.search_url}&page={page + 1}")
            soup_response = BeautifulSoup(response.text, "html.parser")
            ustawy = soup_response.find_all("div", class_="SearchResult")

            ustawy_in_force = [
                u
                for u in ustawy
                if any(
                    p.get_text(strip=True) == "In force"
                    for p in u.find_all("p", class_="forceIndicator")
                )
            ]

            for ustawa in ustawy_in_force:
                link = ustawa.find_all("a", class_="piwik_download")
                if len(link) < 2 or "HTML" not in link[1].get("href", ""):
                    continue

                href = link[1].get("href", "")
                parsed = self._get_html_content(href)
                if parsed:
                    self.all_documents.append(parsed)

    def _get_html_content(self, url):
        response = requests.get(url.replace(".", "https://eur-lex.europa.eu"))
        soup = BeautifulSoup(response.text, "html.parser")

        title_div = soup.find("div", class_="eli-main-title")
        title_parts = title_div.find_all("p", class_="oj-doc-ti") if title_div else []
        plain_text = soup.find("div", id="TexteOnly")
        all_p = soup.find_all("p")
        doc_titles = [p for p in all_p if "doc-ti" in p.get("class", [])]
        articles = [p for p in all_p if "ti-art" in p.get("class", [])]
        group_headers = [p for p in all_p if "oj-ti-grseq-1" in p.get("class", [])]
        subdivisions = soup.find_all(
            "div", id=re.compile(r"^rct_"), class_="eli-subdivision"
        )

        if title_div and title_parts and subdivisions:
            print("Using first format")
            parsed = parse_template_1_first_format(soup, title_parts, subdivisions)
        elif plain_text:
            print("Using second format")
            parsed = parse_template_2_second_format(soup, plain_text)
        elif doc_titles and articles:
            print("Using third format")
            parsed = parse_template_3_third_format(soup, doc_titles)
        elif title_div and title_parts and group_headers and not subdivisions:
            print("Using fourth format")
            parsed = parse_template_4_fourth_format(soup, title_parts, group_headers)
        else:
            print("Unknown document format")
            return []

        # Split long items
        return self._split_if_needed(parsed)

    def _split_if_needed(self, parsed_law, max_len=10000):
        result = []

        def find_split_index(text, breakpoints):
            # Binary search to find the last breakpoint where utf-8 encoded length ≤ max_len
            left, right = 0, len(breakpoints) - 1
            best = None
            while left <= right:
                mid = (left + right) // 2
                candidate = text[: breakpoints[mid]]
                if len(candidate.encode("utf-8")) <= max_len:
                    best = breakpoints[mid]
                    left = mid + 1
                else:
                    right = mid - 1
            return best

        for section in parsed_law:
            name = section.get("name", "")
            text = section.get("text", "")

            if len(text.encode("utf-8")) <= max_len:
                result.append({"name": name, "text": text})
                continue

            print(
                f"Splitting section from law '{name}' (length {len(text.encode('utf-8'))} bytes)..."
            )

            start = 0
            while start < len(text):
                remaining_text = text[start:]
                if len(remaining_text.encode("utf-8")) <= max_len:
                    result.append({"name": name, "text": remaining_text.strip()})
                    break

                # First try newline split
                newline_indices = [
                    i + 1 for i, c in enumerate(remaining_text) if c == "\n"
                ]
                split_at = find_split_index(remaining_text, newline_indices)

                # If no good newline found, try spaces
                if split_at is None:
                    space_indices = [
                        i + 1 for i, c in enumerate(remaining_text) if c == " "
                    ]
                    split_at = find_split_index(remaining_text, space_indices)

                # If no break point found, fallback to brute cutoff
                if split_at is None:
                    # Brute force max-length safe cutoff
                    end = 0
                    while end < len(remaining_text):
                        if len(remaining_text[:end].encode("utf-8")) > max_len:
                            break
                        end += 1
                    split_at = end - 1

                chunk = remaining_text[:split_at].strip()
                result.append({"name": name, "text": chunk})
                start += split_at

        # Final validation
        for i, entry in enumerate(result):
            byte_len = len(entry["text"].encode("utf-8"))
            if byte_len > max_len:
                print(
                    f"Chunk {i} too long after split: {byte_len} bytes — name: {entry['name'][:60]}..."
                )

        return result


if __name__ == "__main__":
    args = parse_cli_args()

    # Assign default path if missing
    if args.source == "json" and not args.path_or_url:
        args.path_or_url = DEFAULT_SAVE_FILE
    elif args.source == "web" and not args.path_or_url:
        args.path_or_url = DEFAULT_EURLEX_URL

    if args.source == "web":
        downloader = EurlexDownloader(args.path_or_url)
        data = downloader()
        save_path = (
            args.save
            if args.save
            else DEFAULT_SAVE_FILE
            if args.save is not None
            else None
        )
        if save_path:
            downloader.save_to_json(data, save_path)
    elif args.source == "json":
        downloader = EurlexDownloader("")
        data = downloader.load_from_json(args.path_or_url)

    print(f"Downloaded {len(data)} documents.")
