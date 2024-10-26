import copy
import re
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, cast

import numpy as np
from langchain_community.utils.math import (
    cosine_similarity,
)
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


def calculate_cosine_distances(sentence_embeddings) -> np.array:
    """Calculate cosine distances between sentences.

    Args:
        sentence_embeddings: List of sentence embeddings to calculate distances for.

    Returns:
        Distance between each pair of adjacent sentences
    """
    return (1 - cosine_similarity(sentence_embeddings, sentence_embeddings)).flatten()[1::len(sentence_embeddings) + 1]


BreakpointThresholdType = Literal["percentile", "standard_deviation", "interquartile"]
BREAKPOINT_DEFAULTS: Dict[BreakpointThresholdType, float] = {
    "percentile": 95,
    "standard_deviation": 3,
    "interquartile": 1.5,
}


class BoundedSemanticChunker(BaseDocumentTransformer):
    """First splits the text using semantic chunking according to the specified
    'breakpoint_threshold_amount', but then uses a RecursiveCharacterTextSplitter
    to split all chunks that are larger than 'max_chunk_size'.

    Adapted from langchain_experimental.text_splitter.SemanticChunker"""

    def __init__(
            self,
            embeddings: Embeddings,
            buffer_size: int = 1,
            add_start_index: bool = False,
            breakpoint_threshold_type: BreakpointThresholdType = "percentile",
            breakpoint_threshold_amount: Optional[float] = None,
            number_of_chunks: Optional[int] = None,
            max_chunk_size: int = 500,
            min_chunk_size: int = 4
    ):
        self._add_start_index = add_start_index
        self.embeddings = embeddings
        self.buffer_size = buffer_size
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.number_of_chunks = number_of_chunks
        if breakpoint_threshold_amount is None:
            self.breakpoint_threshold_amount = BREAKPOINT_DEFAULTS[
                breakpoint_threshold_type
            ]
        else:
            self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        # Splitting the text on '.', '?', and '!'
        self.sentence_split_regex = re.compile(r"(?<=[.?!])\s+")

        assert self.breakpoint_threshold_type == "percentile", "only breakpoint_threshold_type 'percentile' is currently supported"
        assert self.buffer_size == 1, "combining sentences is not supported yet"

    def _calculate_sentence_distances(
        self, sentences: List[dict]
    ) -> Tuple[List[float], List[dict]]:
        """Split text into multiple components."""
        embeddings = self.embeddings.embed_documents(sentences)
        return calculate_cosine_distances(embeddings)

    def _calculate_breakpoint_threshold(self, distances: np.array, alt_breakpoint_threshold_amount=None) -> float:
        if alt_breakpoint_threshold_amount is None:
            breakpoint_threshold_amount = self.breakpoint_threshold_amount
        else:
            breakpoint_threshold_amount = alt_breakpoint_threshold_amount
        if self.breakpoint_threshold_type == "percentile":
            return cast(
                float,
                np.percentile(distances, breakpoint_threshold_amount),
            )
        elif self.breakpoint_threshold_type == "standard_deviation":
            return cast(
                float,
                np.mean(distances)
                + breakpoint_threshold_amount * np.std(distances),
            )
        elif self.breakpoint_threshold_type == "interquartile":
            q1, q3 = np.percentile(distances, [25, 75])
            iqr = q3 - q1

            return np.mean(distances) + breakpoint_threshold_amount * iqr
        else:
            raise ValueError(
                f"Got unexpected `breakpoint_threshold_type`: "
                f"{self.breakpoint_threshold_type}"
            )

    def _threshold_from_clusters(self, distances: List[float]) -> float:
        """
        Calculate the threshold based on the number of chunks.
        Inverse of percentile method.
        """
        if self.number_of_chunks is None:
            raise ValueError(
                "This should never be called if `number_of_chunks` is None."
            )
        x1, y1 = len(distances), 0.0
        x2, y2 = 1.0, 100.0

        x = max(min(self.number_of_chunks, x1), x2)

        # Linear interpolation formula
        y = y1 + ((y2 - y1) / (x2 - x1)) * (x - x1)
        y = min(max(y, 0), 100)

        return cast(float, np.percentile(distances, y))

    def split_text(
        self,
        text: str,
    ) -> List[str]:
        sentences = self.sentence_split_regex.split(text)

        # having len(sentences) == 1 would cause the following
        # np.percentile to fail.
        if len(sentences) == 1:
            return sentences

        bad_sentences = []

        distances = self._calculate_sentence_distances(sentences)

        if self.number_of_chunks is not None:
            breakpoint_distance_threshold = self._threshold_from_clusters(distances)
        else:
            breakpoint_distance_threshold = self._calculate_breakpoint_threshold(
                distances
            )

        indices_above_thresh = [
            i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
        ]

        chunks = []
        start_index = 0

        # Iterate through the breakpoints to slice the sentences
        for index in indices_above_thresh:
            # The end index is the current breakpoint
            end_index = index

            # Slice the sentence_dicts from the current start index to the end index
            group = sentences[start_index : end_index + 1]
            combined_text = " ".join(group)
            if self.min_chunk_size <= len(combined_text) <= self.max_chunk_size:
                chunks.append(combined_text)
            else:
                sent_lengths = np.array([len(sd) for sd in group])
                good_indices = np.flatnonzero(np.cumsum(sent_lengths) <= self.max_chunk_size)
                smaller_group = [group[i] for i in good_indices]
                if smaller_group:
                    combined_text = " ".join(smaller_group)
                    if len(combined_text) >= self.min_chunk_size:
                        chunks.append(combined_text)
                        group = group[good_indices[-1]:]
                bad_sentences.extend(group)

            # Update the start index for the next group
            start_index = index + 1

        # The last group, if any sentences remain
        if start_index < len(sentences):
            group = sentences[start_index:]
            combined_text = " ".join(group)
            if self.min_chunk_size <= len(combined_text) <= self.max_chunk_size:
                chunks.append(combined_text)
            else:
                sent_lengths = np.array([len(sd) for sd in group])
                good_indices = np.flatnonzero(np.cumsum(sent_lengths) <= self.max_chunk_size)
                smaller_group = [group[i] for i in good_indices]
                if smaller_group:
                    combined_text = " ".join(smaller_group)
                    if len(combined_text) >= self.min_chunk_size:
                        chunks.append(combined_text)
                        group = group[good_indices[-1]:]
                bad_sentences.extend(group)

        # If pure semantic chunking wasn't able to split all text,
        # split the remaining problematic text using a recursive character splitter instead
        if len(bad_sentences) > 0:
            recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=self.max_chunk_size, chunk_overlap=10,
                                                                separators=["\n\n", "\n", ".", ", ", " ", ""])
            for bad_sentence in bad_sentences:
                if len(bad_sentence) >= self.min_chunk_size:
                    chunks.extend(recursive_splitter.split_text(bad_sentence))
        return chunks

    def create_documents(
                self, texts: List[str], metadatas: Optional[List[dict]] = None
        ) -> List[Document]:
            """Create documents from a list of texts."""
            _metadatas = metadatas or [{}] * len(texts)
            documents = []
            for i, text in enumerate(texts):
                index = -1
                for chunk in self.split_text(text):
                    metadata = copy.deepcopy(_metadatas[i])
                    if self._add_start_index:
                        index = text.find(chunk, index + 1)
                        metadata["start_index"] = index
                    new_doc = Document(page_content=chunk, metadata=metadata)
                    documents.append(new_doc)
            return documents

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Split documents."""
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)

    def transform_documents(
            self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Transform sequence of documents by splitting them."""
        return self.split_documents(list(documents))



