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
from qdrant_client import QdrantClient, models
from langchain_community.retrievers import QdrantSparseVectorRetriever
from langchain_community.vectorstores.qdrant import Qdrant
from langchain_core.pydantic_v1 import Field
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.schema import Document


def batchify(_list: List, batch_size: int) -> Generator[List, None, None]:
    for i in range(0, len(_list), batch_size):
        yield _list[i:i + batch_size]


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
        vecs = []
        for text_batch in batchify(texts, batch_size):
            with torch.no_grad():
                tokens = self.splade_doc_tokenizer(text_batch, truncation=True, padding=True,
                                                   return_tensors="pt").to(self.device)

                output = self.splade_doc_model(**tokens)
            logits, attention_mask = output.logits, tokens.attention_mask
            relu_log = torch.log(1 + torch.relu(logits))
            weighted_log = relu_log * attention_mask.unsqueeze(-1)
            tvecs, _ = torch.max(weighted_log, dim=1)

            # extract all non-zero values and their indices from the sparse vectors
            for batch in tvecs:
                indices.append(batch.nonzero(as_tuple=True)[0].tolist())
                vecs.append(batch[indices[-1]].tolist())

        return indices, vecs

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
