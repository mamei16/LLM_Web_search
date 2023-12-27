import re
import concurrent.futures

import requests
from bs4 import BeautifulSoup
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.retrievers.document_compressors.embeddings_filter import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import Document
try:
    from langchain_community.retrievers import BM25Retriever
except ImportError:
    BM25Retriever = None


class LangchainCompressor:

    def __init__(self, device="cuda"):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": device})
        self.spaces_regex = re.compile(r" {3,}")

    def preprocess_text(self, text: str) -> str:
        text = text.replace("\n", " \n")
        text = self.spaces_regex.sub(" ", text)
        text = text.strip()
        return text

    def faiss_embedding_query_urls(self, query: str, url_list: list[str], num_results: int = 5,
                                   similarity_threshold: float = 0.5, chunk_size: int = 500) -> list[Document]:
        documents = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(load_url, url): url for url in url_list}
            for future in concurrent.futures.as_completed(future_to_url, timeout=10):
                url = future_to_url[future]
                try:
                    documents.append(future.result())
                except Exception as exc:
                    print('LLM_Web_search | %r generated an exception: %s' % (url, exc))

        if not documents:
            return documents

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=10,
                                                       separators=["\n\n", "\n", ".", ", ", " ", ""])
        texts = text_splitter.split_documents(documents)

        faiss_retriever = FAISS.from_documents(texts, self.embeddings).as_retriever(
            search_kwargs={"k": num_results}
        )
        if not BM25Retriever:
            raise ImportError("Could not import BM25Retriever. Please ensure that you have installed "
                              "langchain==0.0.352")

        #  This sparse retriever is good at finding relevant documents based on keywords,
        #  while the dense retriever is good at finding relevant documents based on semantic similarity.
        bm25_retriever = BM25Retriever.from_documents(texts, preprocess_func=lambda t: t.replace("\n", " \n"))
        bm25_retriever.k = num_results

        redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embeddings)
        embeddings_filter = EmbeddingsFilter(embeddings=self.embeddings, k=None,
                                             similarity_threshold=similarity_threshold)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[redundant_filter, embeddings_filter]
        )
        compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor,
                                                               base_retriever=faiss_retriever)

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, compression_retriever], weights=[0.4, 0.5]
        )

        compressed_docs = ensemble_retriever.get_relevant_documents(query)

        # Ensemble may return more than "num_results" results, so cut off excess ones
        return compressed_docs[:num_results]


def docs_to_pretty_str(docs) -> str:
    ret_str = ""
    for i, doc in enumerate(docs):
        ret_str += f"Result {i+1}:\n"
        ret_str += f"{doc.page_content}\n"
        ret_str += f"Source URL: {doc.metadata['source']}\n\n"
    return ret_str


def load_url(url: str) -> Document:
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
               "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
               "Accept-Language": "en-US,en;q=0.5"}

    response = requests.get(url, headers=headers, verify=True, timeout=8)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if not content_type.startswith("text/html"):
        raise ValueError(f"Expected content type text/html. Got {content_type}.")

    soup = BeautifulSoup(response.content, features="lxml")
    for script in soup(["script", "style"]):
        script.extract()

    strings = '\n'.join([s.strip() for s in soup.stripped_strings])#
    webpage_document = Document(page_content=strings, metadata={"source": url})
    return webpage_document
