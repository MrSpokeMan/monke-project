import requests
from bs4 import BeautifulSoup
import numpy as np


class EurlexDownloader:
    def __init__(self, search_url):
        self.search_url = search_url
        self.all_ustawy = []
        
    def __call__(self):
        self.all_ustawy = self.download_eurlex_page()
        # return self.convert_list_to_array()
        return self.all_ustawy
        
    def download_eurlex_page(self) -> list[list[str]]:
        """Get all the HTML content of the active laws from the search page

        Args:
            search_url (str): URL of the search page

        Returns:
            list[list[str]]: List of lists of strings, where each list of strings is the content of a law
        """
        response = requests.get(self.search_url)
        soup_response = BeautifulSoup(response.text, features="html.parser")
        
        ustawy = soup_response.find_all('div', {"class": 'SearchResult'})
        
        all_ustawy = []
        
        for idx, ustawa in enumerate(ustawy):
            active_state = soup_response.find_all('p', {"class": 'forceIndicator'})[idx].get_text()
            if active_state == 'In force':
                link = ustawa.find_all('a', class_='piwik_download')
                print(link[1]['href'])
                if 'HTML' in link[1]['href']:
                    all_ustawy.append(self._get_html_content(link[1]['href']))             
        return all_ustawy
            
    def _get_html_content(self, url) -> list[str]:
        response = requests.get(url.replace('.', 'https://eur-lex.europa.eu'))
        soup_response = BeautifulSoup(response.text, features="html.parser")
        tab = ["a"]
        i = 1
        ustawa = []
        while tab is not None:
            tab = soup_response.find('div', {"class": "eli-subdivision", "id": f"rct_{i}"})
            i += 1
            if tab is not None:
                ustawa.append(tab.find_all('p', {"class": "oj-normal"})[1].get_text())
        
        return ustawa


    def convert_list_to_array(self) -> np.array(np.array(str)):
        """Converts a list of lists of strings to a numpy array of strings

        Returns:
            np.array(np.array(str)): Numpy array of strings
        """
        ret = np.array(self.all_ustawy, dtype=object)
        for i in range(len(ret)):
            ret[i] = np.array(ret[i], dtype=object)
        return ret

if __name__ == "__main__":
    search_url = "https://eur-lex.europa.eu/search.html?lang=pl&text=industry&qid=1742919459451&type=quick&DTS_SUBDOM=LEGISLATION&scope=EURLEX&FM_CODED=REG&page=1"
    down = EurlexDownloader(search_url)
    print(down().shape)