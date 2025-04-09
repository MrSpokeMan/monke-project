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
        pages = 500
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

        new_format = False
        old_format = False


        title_div = soup_response.find('div', {"class": 'eli-main-title'})
        #print(title_div)
        title_parts = title_div.find_all('p', class_='oj-doc-ti') if title_div else []
        subdivisions = soup_response.find_all('div', id=re.compile(r'^rct_'))

        plain_text = soup_response.find('div', {"id": 'TexteOnly'})

        if (title_div != None and title_parts != None and subdivisions != None):
            new_format = True

        if (plain_text != None):
            old_format = True

        points = []
        if new_format:
            name = "".join(p.get_text(strip=True) for p in title_parts)
            subdivisions = soup_response.find_all('div', id=re.compile(r'^rct_'))
            print("new")
            for div in subdivisions:
                paragraphs = div.find_all('p', class_='oj-normal')
                for i, p in enumerate(paragraphs):
                    if i % 2 == 1:

                        points.append({
                            "name": name,  # Make sure `name` is defined earlier
                            "text": p.get_text(strip=True)
                        })
        if old_format:
            name = soup_response.find('strong').get_text(strip=True)
            #print(name)
            print("old")
            articles = []
            current_article = ""
            collecting = False

            for p in plain_text.find_all('p'):
                text = p.get_text(strip=True)
                if text.startswith("Article "):
                    collecting = True
                    if collecting and current_article != "":
                        articles.append(current_article)
                        current_article = ""
                elif collecting and p.get_text(strip=True) != "\n":
                    current_article += p.get_text(strip=True)


                #print(text)

            if collecting and current_article != "":
                articles.append(current_article)

            grouped_articles = ["\n".join(articles[i:i + 4]) for i in range(0, len(articles), 4)]
            #print(grouped_articles)

            for text in grouped_articles:
                points.append({
                    "name": name,  # Make sure `name` is defined earlier
                    "text": text
                })


        if not old_format and not new_format:
            print("exception")
        return points

if __name__ == "__main__":
    search_url = "https://eur-lex.europa.eu/search.html?lang=en&text=industry&qid=1742919459451&type=quick&DTS_SUBDOM=LEGISLATION&scope=EURLEX&FM_CODED=REG"
    down = EurlexDownloader(search_url)
    edu = down()
    print(edu)
    print("Hakuna")
