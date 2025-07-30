import urllib
from urllib.parse import quote_plus
import re
import logging
import html

import requests
from requests.exceptions import JSONDecodeError
from bs4 import BeautifulSoup

try:
    from .retrieval import DocumentRetriever
    from .utils import Document, Generator
except ImportError:
    from retrieval import DocumentRetriever
    from utils import Document, Generator


def perform_web_search(query, max_results=3, timeout=10):
    """Modified version of function from main webUI in modules/web_search.py"""
    try:
        # Use DuckDuckGo HTML search endpoint
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

        response = requests.get(search_url, headers=headers, timeout=timeout)
        response.raise_for_status()

        # Extract results with regex
        titles = re.findall(r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*>(.*?)</a>', response.text, re.DOTALL)
        urls = re.findall(r'<a[^>]*class="[^"]*result__url[^"]*"[^>]*>(.*?)</a>', response.text, re.DOTALL)
        snippets = re.findall(r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>', response.text, re.DOTALL)

        result_dicts = []
        for i in range(min(len(titles), len(urls), len(snippets), max_results)):
            url = f"https://{urls[i].strip()}"
            title = re.sub(r'<[^>]+>', '', titles[i]).strip()
            title = html.unescape(title)
            snippet = html.unescape(snippets[i]).replace("<b>", "").replace("</b>", "")
            result_dicts.append({"url": url, "title": title, "content": snippet})
        return result_dicts

    except Exception as e:
        logger = logging.getLogger('text-generation-webui')
        logger.error(f"Error performing web search: {e}")
        return []


def retrieve_from_duckduckgo(query: str, document_retriever: DocumentRetriever, max_results: int,
                             simple_search: bool = False):
    documents = []
    query = query.strip("\"'")
    yield f'Getting results from DuckDuckGo...'

    result_documents = []
    result_urls = []
    for result in perform_web_search(query, max_results=max_results):
        result_document = Document(page_content=f"Title: {result['title']}\n{result['content']}",
                                   metadata={"source": result["url"]})
        result_documents.append(result_document)
        result_urls.append(result["url"])

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
