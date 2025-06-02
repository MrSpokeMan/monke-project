import requests
from bs4 import BeautifulSoup
import numpy as np
import re
import json
import os
from template_parser import (
    parse_template_1_first_format,
    parse_template_2_second_format,
    parse_template_3_third_format,
    parse_template_4_fourth_format
)
from cli_utils import parse_cli_args, DEFAULT_EURLEX_URL, DEFAULT_SAVE_FILE


class EurlexDownloader:
    def __init__(self, search_url):
        self.search_url = search_url
        self.all_ustawy = []

    def __call__(self):
        self.download_eurlex_page()
        return self.all_ustawy

    def save_to_json(self, data, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved scraped data to {path}")

    def load_from_json(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No such JSON file: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded data from {path}")
        return data

    def get_last_page_number(self):
        response = requests.get(self.search_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        last_page_link = soup.find('a', title='Last Page')

        if last_page_link:
            href = last_page_link.get('href', '')
            match = re.search(r'page=(\d+)', href)
            if match:
                return int(match.group(1))
        return 1

    def download_eurlex_page(self) -> list[list[dict]]:
        pages = self.get_last_page_number()
        start_page = 0
        end_page = 10
        print(end_page - start_page)
        for page in range(start_page, end_page):
            response = requests.get(self.search_url + f"&page={page + 1}")
            print(f"Fetching: {self.search_url}&page={page + 1}")
            soup_response = BeautifulSoup(response.text, "html.parser")
            ustawy = soup_response.find_all('div', class_='SearchResult')

            ustawy_in_force = [
                u for u in ustawy
                if any(p.get_text(strip=True) == "In force"
                       for p in u.find_all('p', class_='forceIndicator'))
            ]

            for ustawa in ustawy_in_force:
                link = ustawa.find_all('a', class_='piwik_download')
                if len(link) < 2 or 'HTML' not in link[1].get('href', ''):
                    continue

                href = link[1].get('href', '')
                parsed = self._get_html_content(href)
                if parsed:
                    self.all_ustawy.append(parsed)

    def _get_html_content(self, url):
        response = requests.get(url.replace('.', 'https://eur-lex.europa.eu'))
        soup = BeautifulSoup(response.text, 'html.parser')

        title_div = soup.find('div', class_='eli-main-title')
        title_parts = title_div.find_all('p', class_='oj-doc-ti') if title_div else []
        plain_text = soup.find('div', id='TexteOnly')
        all_p = soup.find_all('p')
        doc_titles = [p for p in all_p if 'doc-ti' in p.get('class', [])]
        articles = [p for p in all_p if 'ti-art' in p.get('class', [])]
        group_headers = [p for p in all_p if 'oj-ti-grseq-1' in p.get('class', [])]
        subdivisions = soup.find_all('div', id=re.compile(r'^rct_'), class_='eli-subdivision')

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

    def _split_if_needed(self, parsed_items, max_len=10000):
        new_items = []
        for item in parsed_items:
            text = item.get("text", "")
            name = item.get("name", "")

            if len(text.encode("utf-8")) <= max_len:
                new_items.append(item)
                continue

            print(f"Splitting '{name}' (length {len(text.encode('utf-8'))} bytes)...")

            sentences = text.split('\n')
            chunk = ""
            for sentence in sentences:
                if len((chunk + sentence + '\n').encode('utf-8')) <= max_len:
                    chunk += sentence + '\n'
                else:
                    new_items.append({"name": name, "text": chunk.strip()})
                    chunk = sentence + '\n'

            if chunk.strip():
                new_items.append({"name": name, "text": chunk.strip()})

        return new_items



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
        save_path = args.save if args.save else DEFAULT_SAVE_FILE if args.save is not None else None
        if save_path:
            downloader.save_to_json(data, save_path)
    elif args.source == "json":
        downloader = EurlexDownloader("")
        data = downloader.load_from_json(args.path_or_url)

    print(f"Downloaded {len(data)} documents.")
