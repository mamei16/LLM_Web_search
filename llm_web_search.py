import requests
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup

from .langchain_websearch import faiss_embedding_query_urls, docs_to_pretty_str


def dict_list_to_pretty_str(data: list[dict]) -> str:
    ret_str = ""
    if isinstance(data, dict):
        data = [data]
    if isinstance(data, list):
        for i, d in enumerate(data):
            ret_str += f"Result {i+1}\n"
            ret_str += f"Title: {d['title']}\n"
            ret_str += f"{d['body']}\n"
            ret_str += f"Source URL: {d['href']}\n"
        return ret_str
    else:
        raise ValueError("Input must be dict or list[dict]")


def search_duckduckgo(query: str, max_results: int, instant_answers: bool = True,
                      regular_search_queries: bool = True, get_website_content: bool = False) -> list[dict]:
    with DDGS() as ddgs:
        if instant_answers:
            answer_list = list(ddgs.answers(query))
        else:
            answer_list = None
        if answer_list:
            answer_dict = answer_list[0]
            answer_dict["title"] = query
            answer_dict["body"] = answer_dict["text"]
            answer_dict["href"] = answer_dict["url"]
            answer_dict.pop('icon', None)
            answer_dict.pop('topic', None)
            answer_dict.pop('text', None)
            answer_dict.pop('url', None)
            return [answer_dict]
        elif regular_search_queries:
            results = []
            for result in ddgs.text(query, region='wt-wt', safesearch='moderate', timelimit='y', max_results=max_results):
                if get_website_content:
                    result["body"] = get_webpage_content(result["href"])
                results.append(result)
            return results
        else:
            raise ValueError("One of ('instant_answers', 'regular_search_queries') must be True")


def langchain_search_duckduckgo(query: str, max_results: int, similarity_threshold: float):
    with DDGS() as ddgs:
        result_urls = []
        for result in ddgs.text(query, region='wt-wt', safesearch='moderate', timelimit='y', max_results=max_results):
            result_urls.append(result["href"])
    return docs_to_pretty_str(faiss_embedding_query_urls(query, result_urls, similarity_threshold=similarity_threshold))


def get_webpage_content(url: str) -> str:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.content, 'html.parser')
    content = ""
    for p in soup.find_all('p'):
        content += p.get_text() + "\n"
    return content

