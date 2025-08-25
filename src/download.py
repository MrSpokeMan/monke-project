import asyncio
import json
import logging
import re

import aiohttp
from bs4 import BeautifulSoup

from template_parser import (
    parse_template_1_first_format,
    parse_template_2_second_format,
    parse_template_3_third_format,
    parse_template_4_fourth_format,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EurlexDownloader:
    def __init__(self, search_url: str):
        self.search_url = search_url
        self.all_documents: list[list[dict[str, str]]] = []

    async def __call__(self):
        await self.download_eurlex_page()
        return self.all_documents

    def save_to_json(self, data, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved scraped data to {path}")

    async def get_last_page_number(self, session: aiohttp.ClientSession) -> int:
        async with session.get(self.search_url) as response:
            text = await response.text()
            soup = BeautifulSoup(text, "html.parser")
            last_page_link = soup.find("a", title="Last Page")

            if last_page_link:
                href = last_page_link.get("href", "")
                match = re.search(r"page=(\d+)", href)
                if match:
                    return int(match.group(1))
            return 1

    async def download_eurlex_page(self) -> None:
        async with aiohttp.ClientSession() as session:
            start_page = 0
            end_page = await self.get_last_page_number(session)
            logger.info(f"Total pages to download: {end_page - start_page}")

            batch_size = 50
            all_tasks = []
            for page in range(start_page, end_page):
                page_url = f"{self.search_url}&page={page + 1}"
                task = self.download_single_page(session, page_url, page + 1)
                all_tasks.append(task)

            # Process tasks in batches
            for i in range(0, len(all_tasks), batch_size):
                batch = all_tasks[i : i + batch_size]
                logger.info(f"Processing batch {i // batch_size + 1}")
                batch_results = await asyncio.gather(*batch, return_exceptions=True)

                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error downloading page {i + j + 1}: {result}")
                    elif result:
                        self.all_documents.extend(result)  # type: ignore[arg-type]

    async def download_single_page(
        self, session: aiohttp.ClientSession, page_url: str, page_num: int
    ) -> list[list[dict[str, str]]]:
        try:
            logger.debug(f"Fetching page {page_num}: {page_url}")
            async with session.get(page_url) as response:
                text = await response.text()
                soup_response = BeautifulSoup(text, "html.parser")
                laws = soup_response.find_all("div", class_="SearchResult")

                laws_in_force = [
                    u
                    for u in laws
                    if any(p.get_text(strip=True) == "In force" for p in u.find_all("p", class_="forceIndicator"))
                ]

                html_tasks = []
                for law in laws_in_force:
                    link = law.find_all("a", class_="piwik_download")
                    if len(link) < 2 or "HTML" not in link[1].get("href", ""):
                        continue

                    href = link[1].get("href", "")
                    task = self._get_html_content(session, href)
                    html_tasks.append(task)

                if html_tasks:
                    html_results = await asyncio.gather(*html_tasks, return_exceptions=True)

                    page_documents = []
                    for result in html_results:
                        if isinstance(result, Exception):
                            logger.error(f"Error downloading HTML content: {result}")
                        elif result:
                            page_documents.append(result)

                    return page_documents  # type: ignore[return-value]

                return []

        except Exception as e:
            logger.error(f"Error downloading page {page_num}: {e}")
            return []

    async def _get_html_content(self, session: aiohttp.ClientSession, url: str):
        try:
            full_url = url.replace(".", "https://eur-lex.europa.eu")
            async with session.get(full_url) as response:
                text = await response.text()
                soup = BeautifulSoup(text, "html.parser")

                title_div = soup.find("div", class_="eli-main-title")
                title_parts = title_div.find_all("p", class_="oj-doc-ti") if title_div else []
                plain_text = soup.find("div", id="TexteOnly")
                all_p = soup.find_all("p")
                doc_titles = [p for p in all_p if "doc-ti" in p.get("class", [])]
                articles = [p for p in all_p if "ti-art" in p.get("class", [])]
                group_headers = [p for p in all_p if "oj-ti-grseq-1" in p.get("class", [])]
                subdivisions = soup.find_all("div", id=re.compile(r"^rct_"), class_="eli-subdivision")

                if title_div and title_parts and subdivisions:
                    logger.debug("Using first format")
                    parsed = parse_template_1_first_format(soup, title_parts, subdivisions)
                elif plain_text:
                    logger.debug("Using second format")
                    parsed = parse_template_2_second_format(soup, plain_text)
                elif doc_titles and articles:
                    logger.debug("Using third format")
                    parsed = parse_template_3_third_format(soup, doc_titles)
                elif title_div and title_parts and group_headers and not subdivisions:
                    logger.debug("Using fourth format")
                    parsed = parse_template_4_fourth_format(soup, title_parts, group_headers)
                else:
                    logger.warning("Unknown document format")
                    return []

                return self._split_if_needed(parsed)

        except Exception as e:
            logger.error(f"Error getting HTML content from {url}: {e}")
            return []

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

            logger.debug(f"Splitting section from law '{name}' (length {len(text.encode('utf-8'))} bytes)...")

            start = 0
            while start < len(text):
                remaining_text = text[start:]
                if len(remaining_text.encode("utf-8")) <= max_len:
                    result.append({"name": name, "text": remaining_text.strip()})
                    break

                newline_indices = [i + 1 for i, c in enumerate(remaining_text) if c == "\n"]
                split_at = find_split_index(remaining_text, newline_indices)

                if split_at is None:
                    space_indices = [i + 1 for i, c in enumerate(remaining_text) if c == " "]
                    split_at = find_split_index(remaining_text, space_indices)

                if split_at is None:
                    end = 0
                    while end < len(remaining_text):
                        if len(remaining_text[:end].encode("utf-8")) > max_len:
                            break
                        end += 1
                    split_at = end - 1

                chunk = remaining_text[:split_at].strip()
                result.append({"name": name, "text": chunk})
                start += split_at

        for i, entry in enumerate(result):
            byte_len = len(entry["text"].encode("utf-8"))
            if byte_len > max_len:
                logger.warning(f"Chunk {i} too long after split: {byte_len} bytes — name: {entry['name'][:60]}...")

        return result
