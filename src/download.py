import requests
import os
from bs4 import BeautifulSoup

def download_eurlex_page(search_url, output_folder) -> list[list[str]]:
    """
    Downloads the raw HTML of a searched EUR-Lex page.
    
    :param search_url: URL of the EUR-Lex search page
    :param output_folder: Folder to save downloaded HTML page
    """
    response = requests.get(search_url)
    soup_response = BeautifulSoup(response.text, features="html.parser")
    
    ustawy = soup_response.find_all('div', {"class": 'SearchResult'})
    
    all_ustawy = []
    
    for ustawa in ustawy:
        active_state = soup_response.find('p', {"class": 'forceIndicator'}).get_text()
        if active_state == 'In force':
            link = ustawa.find_all('a', class_='piwik_download')
            print(link[1]['href'])
            if 'HTML' in link[1]['href']:
                all_ustawy.append(get_html_content(link[1]['href']))
                
    return all_ustawy
            
def get_html_content(url) -> list[str]:
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
            

    
# Example usage
if __name__ == "__main__":
    search_url = "https://eur-lex.europa.eu/search.html?lang=pl&text=industry&qid=1742919459451&type=quick&DTS_SUBDOM=LEGISLATION&scope=EURLEX&FM_CODED=REG&page=1"
    output_directory = "eurlex_pages"
    download_eurlex_page(search_url, output_directory)