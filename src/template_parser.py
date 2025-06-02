import re
from bs4 import BeautifulSoup


def parse_template_1_first_format(soup: BeautifulSoup, title_parts, subdivisions) -> list[dict]:
    """Parse using the first (new) format with title_parts and subdivisions."""
    #print("Using first format")
    points = []
    name = "".join(p.get_text(strip=True) for p in title_parts)

    for div in subdivisions:
        paragraphs = div.find_all('p', class_='oj-normal')
        for i, p in enumerate(paragraphs):
            if i % 2 == 1:  # Assume odd paragraphs are the main content
                points.append({
                    "name": name,
                    "text": p.get_text(strip=True)
                })

    return points


def parse_template_2_second_format(soup: BeautifulSoup, plain_text) -> list[dict]:
    """Parse using the second (old) format based on plain_text."""
    #print("Using second format")
    points = []

    name_tag = soup.find('strong')
    name = name_tag.get_text(strip=True) if name_tag else "Unnamed Law"

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

    return points
def parse_template_3_third_format(soup: BeautifulSoup, doc_title_parts) -> list[dict]:
    """Parse using the third format (structured by 'ti-art' and 'normal' paragraph classes)."""
    #print("Using third format")
    points = []

    name = " ".join(p.get_text(strip=True) for p in doc_title_parts)

    all_paragraphs = soup.find_all('p')
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
            collecting = True
            current_article_text = ""
        elif collecting and "normal" in classes:
            current_article_text += p.get_text(strip=True) + " "
        elif collecting and "normal" not in classes:
            if current_article_text.strip():
                points.append({
                    "name": name,
                    "text": current_article_text.strip()
                })
            collecting = False
            current_article_text = ""

    if collecting and current_article_text.strip():
        points.append({
            "name": name,
            "text": current_article_text.strip()
        })

    return points

def parse_template_4_fourth_format(soup: BeautifulSoup, title_parts, group_headers) -> list[dict]:
    """Parse using the fourth format (grouped sections with tables and 'oj-ti-grseq-1' headers)."""
    #print("Using fourth format")
    points = []

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

    return points