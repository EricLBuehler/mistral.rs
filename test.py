from bs4 import BeautifulSoup
from bs4.element import Comment
import requests


def tag_visible(element):
    if element.parent.name in [
        "style",
        "script",
        "head",
        "title",
        "meta",
        "[document]",
    ]:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, "html.parser")
    body_content = soup.find("body")
    if body_content:
        texts = body_content.find_all(text=True)
        visible_texts = filter(tag_visible, texts)
        return " ".join(t.strip() for t in visible_texts)
    return ""


url = "https://www.attitash.com/The%20Mountain/Mountain%20Conditions/Snow%20and%20Weather%20Report.aspx"
# url = "https://github.com/EricLBuehler/mistral.rs/pull/1238"
response = requests.get(url)
if response.status_code == 200:
    visible_text = text_from_html(response.content)
    print(visible_text)
else:
    print("Failed to retrieve the webpage")
