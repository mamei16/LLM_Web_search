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


class SimilarLengthsBatchifyer:
    """
    Generator class to split samples into batches. Groups sample sequences
    of equal/similar length together to minimize the need for padding within a batch.
    """
    def __init__(self, batch_size, inputs, max_padding_len=10):
        # Remember number of samples
        self.num_samples = len(inputs)

        self.unique_lengths = set()
        self.length_to_sample_indices = {}

        for i in range(0, len(inputs)):
            len_input = len(inputs[i])

            self.unique_lengths.add(len_input)

            # For each length, keep track of the indices of the samples that have this length
            # E.g.: self.length_to_sample_indices = { 3: [3,5,11], 4: [1,2], ...}
            if len_input in self.length_to_sample_indices:
                self.length_to_sample_indices[len_input].append(i)
            else:
                self.length_to_sample_indices[len_input] = [i]

        # Use a dynamic batch size to speed up inference at a constant VRAM usage
        self.unique_lengths = sorted(list(self.unique_lengths))
        max_chars_per_batch = self.unique_lengths[-1] * batch_size
        self.length_to_batch_size = {length: int(max_chars_per_batch / (length * batch_size)) * batch_size for length in self.unique_lengths}

        # Merge samples of similar lengths in those cases where the amount of samples
        # of a particular length is < dynamic batch size
        accum_len_diff = 0
        for i in range(1, len(self.unique_lengths)):
            if accum_len_diff >= max_padding_len:
                accum_len_diff = 0
                continue
            curr_len = self.unique_lengths[i]
            prev_len = self.unique_lengths[i-1]
            len_diff = curr_len - prev_len
            if (len_diff <= max_padding_len and
                    (len(self.length_to_sample_indices[curr_len]) < self.length_to_batch_size[curr_len]
                     or len(self.length_to_sample_indices[prev_len]) < self.length_to_batch_size[prev_len])):
                self.length_to_sample_indices[curr_len].extend(self.length_to_sample_indices[prev_len])
                self.length_to_sample_indices[prev_len] = []
                accum_len_diff += len_diff
            else:
                accum_len_diff = 0

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        # Iterate over all possible sentence lengths
        for length in self.unique_lengths:

            # Get indices of all samples for the current length
            # for example, all indices of samples with a length of 7
            sequence_indices = self.length_to_sample_indices[length]
            if len(sequence_indices) == 0:
                continue

            dyn_batch_size = self.length_to_batch_size[length]

            # Compute the number of batches
            num_batches = np.ceil(len(sequence_indices) / dyn_batch_size)

            # Loop over all possible batches
            for batch_indices in np.array_split(sequence_indices, num_batches):
                yield batch_indices


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
        batchifyer = SimilarLengthsBatchifyer(batch_size, texts)
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
            for batch in tvecs.cpu():
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
