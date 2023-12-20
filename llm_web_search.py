import requests
from requests.exceptions import JSONDecodeError
import urllib

from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from langchain.schema import Document

from .langchain_websearch import docs_to_pretty_str, LangchainCompressor


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
            for result in ddgs.text(query, region='wt-wt', safesearch='moderate',
                                    timelimit=None, max_results=max_results):
                if get_website_content:
                    result["body"] = get_webpage_content(result["href"])
                results.append(result)
            return results
        else:
            raise ValueError("One of ('instant_answers', 'regular_search_queries') must be True")


def langchain_search_duckduckgo(query: str, langchain_compressor: LangchainCompressor,
                                max_results: int, similarity_threshold: float, instant_answers: bool):
    documents = []
    with DDGS() as ddgs:
        if instant_answers:
            answer_list = list(ddgs.answers(query))
            if answer_list:
                answer_dict = answer_list[0]
                instant_answer_doc = Document(page_content=answer_dict["text"],
                                              metadata={"source": answer_dict["url"]})
                documents.append(instant_answer_doc)

        results = []
        result_urls = []
        for result in ddgs.text(query, region='wt-wt', safesearch='moderate', timelimit=None, max_results=max_results):
            results.append(result)
            result_urls.append(result["href"])

    documents.extend(langchain_compressor.faiss_embedding_query_urls(query, result_urls,
                                                                     num_results=max_results,
                                                                     similarity_threshold=similarity_threshold))
    if not documents:    # Fall back to old simple search rather than returning nothing
        print("LLM_Web_search | Could not find any page content "
              "similar enough to be extracted, using basic search fallback...")
        return dict_list_to_pretty_str(results)
    return docs_to_pretty_str(documents)


def langchain_search_searxng(query: str, url: str, langchain_compressor: LangchainCompressor,
                             max_results: int, similarity_threshold: float):
    request_str = f"/search?q={urllib.parse.quote(query)}&format=json"
    response = requests.get(url+request_str)
    response.raise_for_status()
    try:
        result_dict = response.json()
    except JSONDecodeError:
        raise ValueError("JSONDecodeError: Please ensure that the SearXNG instance can return data in JSON format")
    result_urls = []
    for result in result_dict["results"]:
        result_urls.append(result["url"])
    documents = langchain_compressor.faiss_embedding_query_urls(query, result_urls,
                                                                num_results=max_results,
                                                                similarity_threshold=similarity_threshold)
    return docs_to_pretty_str(documents)


def get_webpage_content(url: str) -> str:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.content, 'html.parser')
    content = ""
    for p in soup.find_all('p'):
        content += p.get_text() + "\n"
    return content

