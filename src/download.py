import requests
from bs4 import BeautifulSoup
import numpy as np
import re


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
        pages = 10
        print(pages)
        for page in range(pages):
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
        #        print(link[1]['href'])
                if 'HTML' in link[1]['href']:
                    tab = self._get_html_content(link[1]['href'])
                    if (tab != []):
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

        # --- New format parsing ---
        if new_format:
            print("new")
            name = "".join(p.get_text(strip=True) for p in title_parts)
            for div in subdivisions:
                paragraphs = div.find_all('p', class_='oj-normal')
                for i, p in enumerate(paragraphs):
                    if i % 2 == 1:
                        points.append({
                            "name": name,
                            "text": p.get_text(strip=True)
                        })

        # --- Old format parsing ---
        if old_format:
            print("old")
            name = soup_response.find('strong').get_text(strip=True) if soup_response.find('strong') else "Unnamed Law"
            articles = []
            current_article = ""
            collecting = False

            for p in plain_text.find_all('p'):
                text = p.get_text(strip=True)
                if text.startswith("Article "):
                    if collecting and current_article:
                        articles.append(current_article)
                    collecting = True
                    current_article = text + "\n"
                elif collecting and text:
                    current_article += text + "\n"

            if collecting and current_article:
                articles.append(current_article)

            grouped_articles = ["\n".join(articles[i:i + 4]) for i in range(0, len(articles), 4)]

            for text in grouped_articles:
                points.append({
                    "name": name,
                    "text": text
                })

        # --- Third format parsing ---
        if third_format:
            print("third")
            name = " ".join(p.get_text(strip=True) for p in doc_title_parts)

            all_paragraphs = soup_response.find_all('p')
            collecting = False
            current_article_text = ""

            for p in all_paragraphs:
                classes = p.get("class", [])
                if "ti-art" in classes:
                    if collecting and current_article_text.strip():
                        points.append({
                            "name": name,
                            "text": current_article_text.strip()
                        })
                    # Start new article section
                    collecting = True
                    current_article_text = ""  # Reset text buffer
                elif collecting and "normal" in classes:
                    current_article_text += p.get_text(strip=True) + " "
                elif collecting and "normal" not in classes:
                    # If any non-normal paragraph breaks the article block
                    if current_article_text.strip():
                        points.append({
                            "name": name,
                            "text": current_article_text.strip()
                        })
                    collecting = False
                    current_article_text = ""

            # Append last article if still collecting
            if collecting and current_article_text.strip():
                points.append({
                    "name": name,
                    "text": current_article_text.strip()
                })

        if fourth_format:
            print("fourth")
            name = "".join(p.get_text(strip=True) for p in title_parts)
            for header in group_headers:
                current = header.find_next_sibling()
                while current:
                    if current.name == 'table':
                        oj_normals = current.find_all('p', class_='oj-normal')
                        combined_text = " ".join(p.get_text(strip=True) for p in oj_normals)
                        if combined_text.strip():
                            points.append({
                                "name": name,
                                "text": combined_text.strip()
                            })
                    elif current.name == 'p' and 'oj-ti-grseq-1' in current.get('class', []):
                        break
                    current = current.find_next_sibling()

        if not (new_format or old_format or third_format or fourth_format):
            print("exception: unknown format")

        return points

if __name__ == "__main__":
    search_url = "https://eur-lex.europa.eu/search.html?lang=en&text=industry&qid=1742919459451&type=quick&DTS_SUBDOM=LEGISLATION&scope=EURLEX&FM_CODED=REG"
    down = EurlexDownloader(search_url)
    edu = down()
    #print(edu)
    print("Hakuna")
