import re
import asyncio
import warnings

import aiohttp
import requests
from bs4 import BeautifulSoup
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors.embeddings_filter import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_community.retrievers import BM25Retriever
from transformers import AutoTokenizer, AutoModelForMaskedLM
try:
    from qdrant_client import QdrantClient, models
except ImportError:
    qrant_client = None

from .qdrant_retriever import MyQdrantSparseVectorRetriever
from .semantic_chunker import BoundedSemanticChunker


class LangchainCompressor:

    def __init__(self, device="cuda", num_results: int = 5, similarity_threshold: float = 0.5, chunk_size: int = 500,
                 ensemble_weighting: float = 0.5, splade_batch_size: int = 2, keyword_retriever: str = "bm25",
                 model_cache_dir: str = None, chunking_method: str = "character-based",
                 chunker_breakpoint_threshold_amount: int = 10):
        self.device = device
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": device},
                                                cache_folder=model_cache_dir)
        if keyword_retriever == "splade":
            if "QdrantClient" not in globals():
                raise ImportError("Package qrant_client is missing. Please install it using 'pip install qdrant-client")
            self.splade_doc_tokenizer = AutoTokenizer.from_pretrained("naver/efficient-splade-VI-BT-large-doc",
                                                                      cache_dir=model_cache_dir)
            self.splade_doc_model = AutoModelForMaskedLM.from_pretrained("naver/efficient-splade-VI-BT-large-doc",
                                                                         cache_dir=model_cache_dir).to(self.device)
            self.splade_doc_model.to_bettertransformer()
            self.splade_query_tokenizer = AutoTokenizer.from_pretrained("naver/efficient-splade-VI-BT-large-query",
                                                                        cache_dir=model_cache_dir)
            self.splade_query_model = AutoModelForMaskedLM.from_pretrained("naver/efficient-splade-VI-BT-large-query",
                                                                           cache_dir=model_cache_dir).to(self.device)
            self.splade_query_model.to_bettertransformer()
            self.splade_batch_size = splade_batch_size

        self.spaces_regex = re.compile(r" {3,}")
        self.num_results = num_results
        self.similarity_threshold = similarity_threshold
        self.chunking_method = chunking_method
        self.chunk_size = chunk_size
        self.chunker_breakpoint_threshold_amount = chunker_breakpoint_threshold_amount
        self.ensemble_weighting = ensemble_weighting
        self.keyword_retriever = keyword_retriever

    def preprocess_text(self, text: str) -> str:
        text = text.replace("\n", " \n")
        text = self.spaces_regex.sub(" ", text)
        text = text.strip()
        return text

    def retrieve_documents(self, query: str, url_list: list[str]) -> list[Document]:
        yield "Downloading webpages..."
        html_url_tupls = zip(asyncio.run(async_fetch_urls(url_list)), url_list)
        html_url_tupls = [(content, url) for content, url in html_url_tupls if content is not None]
        if not html_url_tupls:
            return []

        documents = [html_to_plaintext_doc(html, url) for html, url in html_url_tupls]
        if self.chunking_method == "semantic":
            text_splitter = BoundedSemanticChunker(self.embeddings, breakpoint_threshold_type="percentile",
                                                        breakpoint_threshold_amount=self.chunker_breakpoint_threshold_amount,
                                                        max_chunk_size=self.chunk_size)
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=10,
                                                                separators=["\n\n", "\n", ".", ", ", " ", ""])
        yield "Chunking page texts..."
        split_docs = text_splitter.split_documents(documents)
        yield "Retrieving relevant results..."
        # filtered_docs = pipeline_compressor.compress_documents(documents, query)
        faiss_retriever = FAISS.from_documents(split_docs, self.embeddings).as_retriever(
            search_kwargs={"k": self.num_results}
        )

        #  The sparse keyword retriever is good at finding relevant documents based on keywords,
        #  while the dense retriever is good at finding relevant documents based on semantic similarity.
        if self.keyword_retriever == "bm25":
            keyword_retriever = BM25Retriever.from_documents(split_docs, preprocess_func=self.preprocess_text)
            keyword_retriever.k = self.num_results
        elif self.keyword_retriever == "splade":
            client = QdrantClient(location=":memory:")
            collection_name = "sparse_collection"
            vector_name = "sparse_vector"

            client.create_collection(
                collection_name,
                vectors_config={},
                sparse_vectors_config={
                    vector_name: models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False,
                        )
                    )
                },
            )

            keyword_retriever = MyQdrantSparseVectorRetriever(
                splade_doc_tokenizer=self.splade_doc_tokenizer,
                splade_doc_model=self.splade_doc_model,
                splade_query_tokenizer=self.splade_query_tokenizer,
                splade_query_model=self.splade_query_model,
                device=self.device,
                client=client,
                collection_name=collection_name,
                sparse_vector_name=vector_name,
                sparse_encoder=None,
                batch_size=self.splade_batch_size,
                k=self.num_results
            )
            keyword_retriever.add_documents(split_docs)
        else:
            raise ValueError("self.keyword_retriever must be one of ('bm25', 'splade')")

        redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embeddings)
        embeddings_filter = EmbeddingsFilter(embeddings=self.embeddings, k=None,
                                             similarity_threshold=self.similarity_threshold)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[redundant_filter, embeddings_filter]
        )

        compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor,
                                                               base_retriever=faiss_retriever)

        ensemble_retriever = EnsembleRetriever(
            retrievers=[compression_retriever, keyword_retriever],
            weights=[self.ensemble_weighting, 1 - self.ensemble_weighting]
        )
        compressed_docs = ensemble_retriever.get_relevant_documents(query)

        # Ensemble may return more than "num_results" results, so cut off excess ones
        return compressed_docs[:self.num_results]


async def async_download_html(url, headers):
    async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(10)) as session:
        try:
            resp = await session.get(url)
            return await resp.text()
        except UnicodeDecodeError:
            print(
                f"LLM_Web_search | {url} generated an exception: Expected content type text/html. Got {resp.headers['Content-Type']}.")
        except TimeoutError as exc:
            print('LLM_Web_search | %r did not load in time' % url)
        except Exception as exc:
            print('LLM_Web_search | %r generated an exception: %s' % (url, exc))
    return None


async def async_fetch_urls(urls):
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
               "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
               "Accept-Language": "en-US,en;q=0.5"}
    webpages = await asyncio.gather(*[(async_download_html(url, headers)) for url in urls])
    return webpages


def docs_to_pretty_str(docs) -> str:
    ret_str = ""
    for i, doc in enumerate(docs):
        ret_str += f"Result {i+1}:\n"
        ret_str += f"{doc.page_content}\n"
        ret_str += f"Source URL: {doc.metadata['source']}\n\n"
    return ret_str


def download_html(url: str) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
               "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
               "Accept-Language": "en-US,en;q=0.5"}

    response = requests.get(url, headers=headers, verify=True, timeout=8)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if not content_type.startswith("text/html"):
        raise ValueError(f"Expected content type text/html. Got {content_type}.")
    return response.content


def html_to_plaintext_doc(html_text: str or bytes, url: str) -> Document:
    with warnings.catch_warnings(action="ignore"):
        soup = BeautifulSoup(html_text, features="lxml")
    for script in soup(["script", "style"]):
        script.extract()

    strings = '\n'.join([s.strip() for s in soup.stripped_strings])
    webpage_document = Document(page_content=strings, metadata={"source": url})
    return webpage_document
