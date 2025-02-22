from typing import (
    Iterable,
    List,
    Optional,
    Tuple,
    Dict
)

import torch
import numpy as np
from scipy.sparse import csr_array

try:
    from ..utils import Document, SimilarLengthsBatchifyer
except:
    from utils import Document, SimilarLengthsBatchifyer


def neg_dot_dist(x, y):
    dist = np.dot(x, y).data
    if dist.size == 0:  # no overlapping non-zero entries between x and y
        return np.inf
    return -dist.sum()


class SpladeRetriever:
    def __init__(self, splade_doc_tokenizer, splade_doc_model, splade_query_tokenizer, splade_query_model,
                 device, batch_size, k):
        self.splade_doc_tokenizer = splade_doc_tokenizer
        self.splade_doc_model = splade_doc_model
        self.splade_query_tokenizer = splade_query_tokenizer
        self.splade_query_model = splade_query_model
        self.device = device
        self.batch_size = batch_size
        self.k = k
        self.vocab_size = splade_doc_model.config.vocab_size
        self.texts: List[str] = []
        self.metadatas: List[Dict] = []
        self.sparse_doc_vecs: List[csr_array] = []

    def compute_document_vectors(self, texts: List[str], batch_size: int) -> Tuple[List[List[int]], List[List[float]]]:
        indices = []
        values = []
        tokenized_texts = self.splade_doc_tokenizer(texts, truncation=False, padding=False,
                                                    return_tensors="np")["input_ids"]
        batchifyer = SimilarLengthsBatchifyer(batch_size, tokenized_texts)
        texts = np.array(texts)
        batch_indices = []
        for index_batch in batchifyer:
            batch_indices.append(index_batch)
            with torch.no_grad():
                tokens = self.splade_doc_tokenizer(texts[index_batch].tolist(), truncation=True, padding=True,
                                                   return_tensors="pt").to(self.device)
                output = self.splade_doc_model(**tokens)
            logits, attention_mask = output.logits, tokens.attention_mask
            relu_log = torch.log(1 + torch.relu(logits))
            weighted_log = relu_log * attention_mask.unsqueeze(-1)
            tvecs, _ = torch.max(weighted_log, dim=1)

            # extract all non-zero values and their indices from the sparse vectors
            for batch in tvecs.cpu().to(torch.float32):
                indices.append(batch.nonzero(as_tuple=True)[0].numpy())
                values.append(batch[indices[-1]].numpy())

        # Restore order after SimilarLengthsBatchifyer disrupted it:
        # Ensure that the order of 'indices' and 'values' matches the order of the 'texts' parameter
        batch_indices = np.concatenate(batch_indices)
        sorted_indices = np.argsort(batch_indices)
        indices = [indices[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        return indices, values

    def compute_query_vector(self, text: str):
        """
        Computes a vector from logits and attention mask using ReLU, log, and max operations.
        """
        with torch.no_grad():
            tokens = self.splade_query_tokenizer(text, return_tensors="pt").to(self.device)
            output = self.splade_query_model(**tokens)
        logits, attention_mask = output.logits, tokens.attention_mask
        relu_log = torch.log(1 + torch.relu(logits))
        weighted_log = relu_log * attention_mask.unsqueeze(-1)
        max_val, _ = torch.max(weighted_log, dim=1)
        query_vec = max_val.squeeze().cpu().to(torch.float32)

        query_indices = query_vec.nonzero().numpy().flatten()
        query_values = query_vec.detach().numpy()[query_indices]

        return query_indices, query_values

    def add_documents(self, documents: List[Document])-> List[str]:
        """Run more documents through the embeddings and add to the vectorstore.

        Args:
            documents (List[Document]: Documents to add to the vectorstore.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        if not documents:
            return []
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None):

        # Remove duplicate and empty texts
        text_to_metadata = {texts[i]: metadatas[i] for i in range(len(texts)) if len(texts[i]) > 0}
        texts = list(text_to_metadata.keys())
        metadatas = list(text_to_metadata.values())
        self.texts = texts
        self.metadatas = metadatas

        indices, values = self.compute_document_vectors(texts, self.batch_size)
        self.sparse_doc_vecs = [csr_array((val, (ind,)),
                                          shape=(self.vocab_size,)) for val, ind in zip(values, indices)]

        if self.device == "cuda":
            torch.cuda.empty_cache()

    def get_relevant_documents(self, query: str) -> List[Document]:
        if not self.texts:
            return []
        query_indices, query_values = self.compute_query_vector(query)

        sparse_query_vec = csr_array((query_values, (query_indices,)),shape=(self.vocab_size,))
        dists = [neg_dot_dist(sparse_query_vec, doc_vec) for doc_vec in self.sparse_doc_vecs]
        sorted_indices = np.argsort(dists)

        return [Document(self.texts[i], self.metadatas[i]) for i in sorted_indices[:self.k]]
