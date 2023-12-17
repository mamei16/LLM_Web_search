from typing import List

import faiss
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex, StorageContext
from llama_index.schema import NodeWithScore

from llama_index.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore


class FileFaissRetriever:

    def __init__(self, model):
        self.model = model

    def load_directory(self, dir_path):
        documents = SimpleDirectoryReader(dir_path, recursive=True).load_data()

        text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
        service_context = ServiceContext.from_defaults(text_splitter=text_splitter, llm=None,
                                                       embed_model=embed_model)

        # dimensions of all-MiniLM-L6-v2
        d = 384
        faiss_index = faiss.IndexFlatL2(d)
        vector_store = FaissVectorStore(faiss_index)

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context,
                                                service_context=service_context)
        self.retriever = index.as_retriever()

    def query(self, query: str):
        response_nodes = self.retriever.retrieve(query)
        return self.nodes_to_pretty_str(response_nodes)

    @staticmethod
    def nodes_to_pretty_str(node_list: List[NodeWithScore]) -> str:
        ret_str = ""
        for i, node in enumerate(node_list):
            sub_node = node.node
            meta_data = sub_node.metadata
            ret_str += f"Result {i + 1}:\n"
            ret_str += f"{sub_node.text}\n"
            ret_str += f"Source filepath: {meta_data['file_path']}\n"
            ret_str += f"Last modified: {meta_data['last_modified_date']}\n\n"
        return ret_str


embed_model = HuggingFaceEmbedding(model_name="/home/marme/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2/")

retriever = FileFaissRetriever(embed_model)

retriever.load_directory("/home/marme/Documents/articles")

print(retriever.query("stepwise regression?"))
