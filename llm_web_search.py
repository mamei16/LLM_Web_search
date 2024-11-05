import urllib

import requests
from requests.exceptions import JSONDecodeError
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup

try:
    from .retrieval import DocumentRetriever
    from .utils import Document, Generator
except ImportError:
    from retrieval import DocumentRetriever
    from utils import Document, Generator


def search_duckduckgo(query: str, max_results: int, instant_answers: bool = True,
                      regular_search_queries: bool = True, get_website_content: bool = False) -> list[dict]:
    query = query.strip("\"'")
    with DDGS() as ddgs:
        if instant_answers:
            answer_list = ddgs.answers(query)
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


def retrieve_from_duckduckgo(query: str, document_retriever: DocumentRetriever, max_results: int,
                                instant_answers: bool, simple_search: bool = False):
    documents = []
    query = query.strip("\"'")
    yield f'Getting results from DuckDuckGo...'
    with DDGS() as ddgs:
        if instant_answers:
            answer_list = ddgs.answers(query)
            if answer_list:
                if max_results > 1:
                    max_results -= 1  # We already have 1 result now
                answer_dict = answer_list[0]
                instant_answer_doc = Document(page_content=answer_dict["text"],
                                              metadata={"source": answer_dict["url"]})
                documents.append(instant_answer_doc)

        result_documents = []
        result_urls = []
        for result in ddgs.text(query, region='wt-wt', safesearch='moderate', timelimit=None,
                                max_results=document_retriever.num_results):
            result_document = Document(page_content=f"Title: {result['title']}\n{result['body']}",
                                       metadata={"source": result["href"]})
            result_documents.append(result_document)
            result_urls.append(result["href"])

    if simple_search:
        retrieval_gen = Generator(document_retriever.retrieve_from_snippets(query, result_documents))
    else:
        retrieval_gen = Generator(document_retriever.retrieve_from_webpages(query, result_urls))

    for status_message in retrieval_gen:
        yield status_message
    documents.extend(retrieval_gen.retval)

    if not documents:    # Fall back to old simple search rather than returning nothing
        print("LLM_Web_search | Could not find any page content "
              "similar enough to be extracted, using basic search fallback...")
        return result_documents[:max_results]
    return documents[:max_results]


def retrieve_from_searxng(query: str, url: str, document_retriever: DocumentRetriever, max_results: int,
                          instant_answers: bool, simple_search: bool = False):
    yield f'Getting results from Searxng...'
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
               "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
               "Accept-Language": "en-US,en;q=0.5"}
    result_documents = []
    result_urls = []
    request_str = f"/search?q={urllib.parse.quote(query)}&format=json&pageno="
    pageno = 1
    while len(result_urls) < document_retriever.num_results:
        response = requests.get(url + request_str + str(pageno), headers=headers)

        if not result_urls:     # no results to lose by raising an exception here
            response.raise_for_status()
        try:
            response_dict = response.json()
        except JSONDecodeError:
            raise ValueError("JSONDecodeError: Please ensure that the SearXNG instance can return data in JSON format")

        result_dicts = response_dict["results"]
        if not result_dicts:
            break
        for result in result_dicts:
            if "content" in result:   # Since some websites don't provide any description
                result_document = Document(page_content=f"Title: {result['title']}\n{result['content']}",
                                           metadata={"source": result["url"]})
                result_documents.append(result_document)
            result_urls.append(result["url"])

        answers = response_dict["answers"]
        if instant_answers:
            for answer in answers:
                answer_document = Document(page_content=f"Title: {query}\n{answer}",
                                           metadata={"source": "SearXNG instant answer"})
                result_documents.append(answer_document)
        pageno += 1

    if simple_search:
        retrieval_gen = Generator(document_retriever.retrieve_from_snippets(query, result_documents))
    else:
        retrieval_gen = Generator(document_retriever.retrieve_from_webpages(query, result_urls))

    for status_message in retrieval_gen:
        yield status_message
    documents = retrieval_gen.retval
    return documents[:max_results]


def get_webpage_content(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
               "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
               "Accept-Language": "en-US,en;q=0.5"}
    if not url.startswith("https://"):
        try:
            response = requests.get(f"https://{url}", headers=headers)
        except:
            response = requests.get(url, headers=headers)
    else:
        response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.content, features="lxml")
    for script in soup(["script", "style"]):
        script.extract()

    strings = soup.stripped_strings
    return '\n'.join([s.strip() for s in strings])
