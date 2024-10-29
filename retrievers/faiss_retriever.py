from typing import List, Callable

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from ..utils import Document, cosine_similarity
except:
    from utils import Document, cosine_similarity


class FaissRetriever:

    def __init__(self, embedding_model: SentenceTransformer, num_results: int = 5, similarity_threshold: float = 0.5):
        self.embedding_model = embedding_model
        self.num_results = num_results
        self.similarity_threshold = similarity_threshold
        self.index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())

    def add_documents(self, documents: List[Document]):
        self.documents = documents
        self.document_embeddings = self.embedding_model.encode([doc.page_content for doc in documents])
        self.index.add(self.document_embeddings)

    def get_relevant_documents(self, query: str) -> List[Document]:
        query_embedding = self.embedding_model.encode(query)
        D, I = self.index.search(query_embedding.reshape(1, -1), self.num_results)
        result_indices = I[0]
        relevant_doc_embeddings = self.document_embeddings[result_indices]
        # dense_result_docs = [split_docs[i] for i in I[0]]

        # Filter out redundant documents
        included_idxs = filter_similar_embeddings(relevant_doc_embeddings, cosine_similarity,
                                                  threshold=0.95)
        relevant_doc_embeddings = relevant_doc_embeddings[included_idxs]

        # Filter out documents that aren't similar enough
        similarity = cosine_similarity([query_embedding], relevant_doc_embeddings)[0]
        similar_enough = np.where(similarity > self.similarity_threshold)[0]
        included_idxs = [included_idxs[i] for i in similar_enough]

        filtered_result_indices = result_indices[included_idxs]
        return [self.documents[i] for i in filtered_result_indices]


def filter_similar_embeddings(
    embedded_documents: List[List[float]], similarity_fn: Callable, threshold: float
) -> List[int]:
    """Filter redundant documents based on the similarity of their embeddings."""
    similarity = np.tril(similarity_fn(embedded_documents, embedded_documents), k=-1)
    redundant = np.where(similarity > threshold)
    redundant_stacked = np.column_stack(redundant)
    redundant_sorted = np.argsort(similarity[redundant])[::-1]
    included_idxs = set(range(len(embedded_documents)))
    for first_idx, second_idx in redundant_stacked[redundant_sorted]:
        if first_idx in included_idxs and second_idx in included_idxs:
            # Default to dropping the second document of any highly similar pair.
            included_idxs.remove(second_idx)
    return list(sorted(included_idxs))