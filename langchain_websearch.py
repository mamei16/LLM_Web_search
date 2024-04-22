import re
import concurrent.futures

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
from langchain_community.retrievers import BM25Retriever, QdrantSparseVectorRetriever
from qdrant_client import QdrantClient, models
from transformers import AutoTokenizer, AutoModelForMaskedLM
from .qdrant_retriever import MyQdrantSparseVectorRetriever


class LangchainCompressor:

    def __init__(self, device="cuda", num_results: int = 5, similarity_threshold: float = 0.5, chunk_size: int = 500,
                 ensemble_weighting: float = 0.5, splade_batch_size: int = 2):
        self.device = device
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": device})
        self.splade_doc_tokenizer = AutoTokenizer.from_pretrained("naver/efficient-splade-VI-BT-large-doc")
        self.splade_doc_model = AutoModelForMaskedLM.from_pretrained("naver/efficient-splade-VI-BT-large-doc").to(
            self.device)
        self.splade_query_tokenizer = AutoTokenizer.from_pretrained("naver/efficient-splade-VI-BT-large-query")
        self.splade_query_model = AutoModelForMaskedLM.from_pretrained("naver/efficient-splade-VI-BT-large-query").to(
            self.device)
        self.splade_batch_size = splade_batch_size
        self.spaces_regex = re.compile(r" {3,}")
        self.num_results = num_results
        self.similarity_threshold = similarity_threshold
        self.chunk_size = chunk_size
        self.ensemble_weighting = ensemble_weighting
        self.splade_batch_size = splade_batch_size

    def preprocess_text(self, text: str) -> str:
        text = text.replace("\n", " \n")
        text = self.spaces_regex.sub(" ", text)
        text = text.strip()
        return text

    def retrieve_documents(self, query: str, url_list: list[str]) -> list[Document]:
        html_url_tupls = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(download_html, url): url for url in url_list}
            try:
                for future in concurrent.futures.as_completed(future_to_url, timeout=10):
                    url = future_to_url[future]
                    try:
                        html_url_tupls.append((future.result(), url))
                    except Exception as exc:
                        print('LLM_Web_search | %r generated an exception: %s' % (url, exc))
            except TimeoutError as exc:
                exc_str = str(exc).replace("futures unfinished", "websites did not load in time")
                print(f'LLM_Web_search | {exc_str}')

        if not html_url_tupls:
            return []

        documents = [html_to_plaintext_doc(html, url) for html, url in html_url_tupls]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=10,
                                                       separators=["\n\n", "\n", ".", ", ", " ", ""])
        split_docs = text_splitter.split_documents(documents)
        # filtered_docs = pipeline_compressor.compress_documents(documents, query)
        faiss_retriever = FAISS.from_documents(split_docs, self.embeddings).as_retriever(
            search_kwargs={"k": self.num_results}
        )

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

        # Create a retriever with a demo encoder
        qdrant_retriever = MyQdrantSparseVectorRetriever(
            splade_doc_tokenizer=self.splade_doc_tokenizer,
            splade_doc_model=self.splade_doc_model,
            splade_query_tokenizer = self.splade_query_tokenizer,
            splade_query_model = self.splade_query_model,
            device=self.device,
            client=client,
            collection_name=collection_name,
            sparse_vector_name=vector_name,
            sparse_encoder=None,
            batch_size=self.splade_batch_size
        )

        qdrant_retriever.add_documents(split_docs)


        #  This sparse retriever is good at finding relevant documents based on keywords,
        #  while the dense retriever is good at finding relevant documents based on semantic similarity.
        #bm25_retriever = BM25Retriever.from_documents(split_docs, preprocess_func=self.preprocess_text)
        #bm25_retriever.k = self.num_results

        redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embeddings)
        embeddings_filter = EmbeddingsFilter(embeddings=self.embeddings, k=None,
                                             similarity_threshold=self.similarity_threshold)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[redundant_filter, embeddings_filter]
        )

        compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor,
                                                               base_retriever=faiss_retriever)

        ensemble_retriever = EnsembleRetriever(
            retrievers=[compression_retriever, qdrant_retriever],
            weights=[self.ensemble_weighting, 1 - self.ensemble_weighting]
        )
        compressed_docs = ensemble_retriever.get_relevant_documents(query)

        # Ensemble may return more than "num_results" results, so cut off excess ones
        return compressed_docs[:self.num_results]


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
    soup = BeautifulSoup(html_text, features="lxml")
    for script in soup(["script", "style"]):
        script.extract()

    strings = '\n'.join([s.strip() for s in soup.stripped_strings])
    webpage_document = Document(page_content=strings, metadata={"source": url})
    return webpage_document
