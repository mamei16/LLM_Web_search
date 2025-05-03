from typing import List, Callable

import faiss
import numpy as np

try:
    from ..utils import Document, cosine_similarity, MySentenceTransformer, SimilarLengthsBatchifyer
except:
    from utils import Document, cosine_similarity, MySentenceTransformer, SimilarLengthsBatchifyer


class FaissRetriever:

    def __init__(self, embedding_model: MySentenceTransformer, num_results: int = 5, similarity_threshold: float = 0.5):
        self.embedding_model = embedding_model
        self.num_results = num_results
        self.similarity_threshold = similarity_threshold
        self.index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
        self.document_embeddings = []
        self.documents = []
        self.text_to_embedding = {}

    def add_documents(self, documents: List[Document]):
        if not documents:
            return
        self.documents = documents
        self.document_embeddings = self.embedding_model.batch_encode([doc.page_content for doc in documents])
        self.index.add(self.document_embeddings)
        self.text_to_embedding = {document.page_content: embedding
                                  for document, embedding in zip(documents, self.document_embeddings)}

    def get_relevant_documents(self, query: str) -> List[Document]:
        if not self.documents:
            return []
        query_embedding = self.embedding_model.encode(query)
        D, I = self.index.search(query_embedding.reshape(1, -1), self.num_results)
        result_indices = I[0]
        relevant_doc_embeddings = self.document_embeddings[result_indices]
        # dense_result_docs = [split_docs[i] for i in I[0]]

        # Filter out documents that aren't similar enough
        similarity = cosine_similarity([query_embedding], relevant_doc_embeddings)[0]
        similar_enough = np.where(similarity > self.similarity_threshold)[0]

        filtered_result_indices = result_indices[similar_enough]
        return [self.documents[i] for i in filtered_result_indices]
