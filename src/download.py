import requests
from bs4 import BeautifulSoup
import numpy as np
import re
from template_parser import parse_template_1_first_format, parse_template_2_second_format, parse_template_3_third_format, parse_template_4_fourth_format

class EurlexDownloader:
    def __init__(self, search_url):
        self.search_url = search_url
        self.all_ustawy = []

    def __call__(self):
        self.download_eurlex_page()
        # return self.convert_list_to_array()
        return self.all_ustawy

    def get_last_page_number(self):
        response = requests.get(self.search_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Debugging: check if we can find the "last page" link
        last_page_link = soup.find('a', title='Last Page')

        if last_page_link:
            print("Found 'Last Page' link:", last_page_link)
            href = last_page_link.get('href', '')

            # Debugging: print href to check the URL
            print(f"Last page link href: {href}")

            # Regex to extract the page number from the href link
            match = re.search(r'page=(\d+)', href)

            if match:
                return int(match.group(1))  # Returns the last page number
            else:
                print("Page number not found in the URL")
        else:
            print("No 'Last Page' link found")

        return 1  # Fallback if no last page link is found


    def download_eurlex_page(self) -> list[list[dict]]:
        """Get all the HTML content of the active laws from the search page

        Args:
            search_url (str): URL of the search page

        Returns:
            list[list[str]]: List of lists of strings, where each list of strings is the content of a law
        """

        pages = self.get_last_page_number()
        start_page = 0
        end_page = 10
        print(end_page-start_page)
        for page in range(start_page, end_page):
            response = requests.get(self.search_url + f"&page={page + 1}")
            print(self.search_url + f"&page={page + 1}")
            soup_response = BeautifulSoup(response.text, features="html.parser")



            ustawy = soup_response.find_all('div', {"class": 'SearchResult'})
            #print(ustawy)
            ustawy_in_force = []


            for ustawa in ustawy:
                indicators = ustawa.find_all('p', {"class": 'forceIndicator'})
                for p in indicators:
                    text = p.get_text(strip=True)
                    if (text == "In force"):
                        ustawy_in_force.append(ustawa)

            for ustawa in ustawy_in_force:
                link = ustawa.find_all('a', class_='piwik_download')

                # Ensure there are at least 2 links and the second one is an HTML file
                if len(link) < 2:
                    print("Less than 2 links found, skipping.")
                    continue

                href = link[1].get('href', '')
                if 'HTML' not in href:
                    print("Second link is not an HTML version, skipping.")
                    continue

                # Safe to fetch and parse
                tab = self._get_html_content(href)
                if tab:
                    self.all_ustawy.append(tab)

    def _get_html_content(self, url):
        response = requests.get(url.replace('.', 'https://eur-lex.europa.eu'))
        soup_response = BeautifulSoup(response.text, features="html.parser")

        # Determine format flags
        title_div = soup_response.find('div', {"class": 'eli-main-title'})
        title_parts = title_div.find_all('p', class_='oj-doc-ti') if title_div else []
        plain_text = soup_response.find('div', {"id": 'TexteOnly'})

        # Fetch all relevant paragraphs only once
        all_paragraphs = soup_response.find_all('p')
        doc_title_parts = [p for p in all_paragraphs if 'doc-ti' in p.get('class', [])]
        article_titles = [p for p in all_paragraphs if 'ti-art' in p.get('class', [])]
        group_headers = [p for p in all_paragraphs if 'oj-ti-grseq-1' in p.get('class', [])]

        subdivisions = soup_response.find_all('div', id=re.compile(r'^rct_'), class_='eli-subdivision')

        new_format = title_div and title_parts and subdivisions
        old_format = bool(plain_text)
        third_format = doc_title_parts and article_titles
        fourth_format = title_div and title_parts and group_headers and not subdivisions

        points = []

        if title_div and title_parts and subdivisions:
            print("Using first format")
            parsed = self._split_if_needed(parse_template_1_first_format(soup_response, title_parts, subdivisions))
        elif plain_text:
            print("Using second format")
            parsed = self._split_if_needed(parse_template_2_second_format(soup_response, plain_text))
        elif doc_title_parts and article_titles:
            print("Using third format")
            parsed = self._split_if_needed(parse_template_3_third_format(soup_response, doc_title_parts))
        elif title_div and title_parts and group_headers and not subdivisions:
            print("Using fourth format")
            parsed = self._split_if_needed(parse_template_4_fourth_format(soup_response, title_parts, group_headers))
        else:
            print("exception: unknown format")
            return []

        for item in parsed:
            if len(item.get("text", "")) >= 10000:
                print("Length of text:", len(item.get("text", "")))

        return parsed


    def _split_if_needed(self, parsed_items, max_len=10000):
        new_items = []

        for item in parsed_items:
            text = item.get("text", "")
            name = item.get("name", "")

            if len(text) <= max_len:
                new_items.append(item)
                continue

            print(f"Splitting '{name}' (length {len(text)})...")

            # Split at \n boundaries
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
    search_url = "https://eur-lex.europa.eu/search.html?lang=en&text=industry&qid=1742919459451&type=quick&DTS_SUBDOM=LEGISLATION&scope=EURLEX&FM_CODED=REG"
    down = EurlexDownloader(search_url)
    edu = down()
    #print(edu)
    print("Hakuna")
