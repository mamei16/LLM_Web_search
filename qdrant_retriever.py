from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Tuple,
    cast,
    Generator
)

import torch
from torch.utils.data import Sampler
import numpy as np
from langchain_community.retrievers import QdrantSparseVectorRetriever
from langchain_community.vectorstores.qdrant import Qdrant
from langchain_core.pydantic_v1 import Field
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.schema import Document
try:
    from qdrant_client import QdrantClient, models
except ImportError:
    pass


def batchify(_list: List, batch_size: int) -> Generator[List, None, None]:
    for i in range(0, len(_list), batch_size):
        yield _list[i:i + batch_size]


class EqualLengthsBatchSampler(Sampler):

    def __init__(self, batch_size, inputs):
        # Remember batch size and number of samples
        self.batch_size, self.num_samples = batch_size, len(inputs)

        self.unique_lengths = set()
        self.length_to_samples = {}

        for i in range(0, len(inputs)):
            len_input = len(inputs[i])

            # Add length pair to set of all seen pairs
            self.unique_lengths.add(len_input)

            # For each lengths pair, keep track of which sample indices for this pair
            # E.g.: self.lengths_to_sample = { (4,5): [3,5,11], (5,5): [1,2,9], ...}
            if len_input in self.length_to_samples:
                self.length_to_samples[len_input].append(inputs[i])
            else:
                self.length_to_samples[len_input] = [inputs[i]]

        # Convert set of unique length pairs to a list so we can shuffle it later
        self.unique_lengths = list(self.unique_lengths)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        # Iterate over all possible sentence lengths
        for length in self.unique_lengths:

            # Get indices of all samples for the current length
            # for example, all indices of samples with a length of 7
            sequences = self.length_to_samples[length]
            #sequences = list(sequences)

            # Compute the number of batches
            num_batches = np.ceil(len(sequences) / self.batch_size)

            # Loop over all possible batches
            for batch in np.array_split(sequences, num_batches):
                yield batch.tolist()


class MyQdrantSparseVectorRetriever(QdrantSparseVectorRetriever):
    splade_doc_tokenizer: Any = Field(repr=False)
    splade_doc_model: Any = Field(repr=False)
    splade_query_tokenizer: Any = Field(repr=False)
    splade_query_model: Any = Field(repr=False)
    device: Any = Field(repr=False)
    batch_size: int = Field(repr=False)
    sparse_encoder: Any or None = Field(repr=False)

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def compute_document_vectors(self, texts: List[str], batch_size: int) -> Tuple[List[List[int]], List[List[float]]]:
        indices = []
        values = []
        sampler = EqualLengthsBatchSampler(batch_size, texts)
        for text_batch in sampler:
            with torch.no_grad():
                tokens = self.splade_doc_tokenizer(text_batch, truncation=True, padding=True,
                                                   return_tensors="pt").to(self.device)
                output = self.splade_doc_model(**tokens)
            logits, attention_mask = output.logits, tokens.attention_mask
            relu_log = torch.log(1 + torch.relu(logits))
            weighted_log = relu_log * attention_mask.unsqueeze(-1)
            tvecs, _ = torch.max(weighted_log, dim=1)

            # extract all non-zero values and their indices from the sparse vectors
            for batch in tvecs.cpu():
                indices.append(batch.nonzero(as_tuple=True)[0].numpy())
                values.append(batch[indices[-1]].numpy())

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
        query_vec = max_val.squeeze().cpu()

        query_indices = query_vec.nonzero().numpy().flatten()
        query_values = query_vec.detach().numpy()[query_indices]

        return query_indices, query_values

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ):
        client = cast(QdrantClient, self.client)

        # Remove duplicate texts
        text_to_metadata = {texts[i]: metadatas[i] for i in range(len(texts))}
        texts = list(text_to_metadata.keys())
        metadatas = list(text_to_metadata.values())

        indices, values = self.compute_document_vectors(texts, self.batch_size)

        points = [
            models.PointStruct(
                id=i + 1,
                vector={
                    self.sparse_vector_name: models.SparseVector(
                        indices=indices[i],
                        values=values[i],
                    )
                },
                payload={
                    self.content_payload_key: texts[i],
                    self.metadata_payload_key: metadatas[i] if metadatas else None,
                },
            )
            for i in range(len(texts))
        ]
        client.upsert(self.collection_name, points=points, **kwargs)
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        client = cast(QdrantClient, self.client)
        query_indices, query_values = self.compute_query_vector(query)

        results = client.search(
            self.collection_name,
            query_filter=self.filter,
            query_vector=models.NamedSparseVector(
                name=self.sparse_vector_name,
                vector=models.SparseVector(
                    indices=query_indices,
                    values=query_values,
                ),
            ),
            limit=self.k,
            with_vectors=False,
            **self.search_options,
        )

        return [
            Qdrant._document_from_scored_point(
                point,
                self.collection_name,
                self.content_payload_key,
                self.metadata_payload_key,
            )
            for point in results
        ]
