from typing import List

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
try:
    from ..chunkers.base_chunker import TextSplitter
except:
    from chunkers.base_chunker import TextSplitter


def split_into_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def split_text_into_even_chunks(tokenizer, text):
    ids_plus = tokenizer(text, truncation=False, add_special_tokens=True, return_offsets_mapping=True)
    token_offset_tups = ids_plus["offset_mapping"]
    offset_tup_chunks = list(split_into_chunks(token_offset_tups, n=tokenizer.model_max_length))
    token_chunks = list(split_into_chunks(ids_plus["input_ids"], n=tokenizer.model_max_length))
    return token_chunks, offset_tup_chunks


def split_into_semantic_chunks(text, separator_indices: List[int]):
    start_index = 0

    for idx in separator_indices:
        chunk = text[start_index:idx]
        yield chunk.strip()
        start_index = idx

    if start_index < len(text):
        yield text[start_index:].strip()


class NerChunker(TextSplitter):
    def __init__(self, model_name="mirth/chonky_distilbert_uncased_1", device="cpu", model_cache_dir: str = None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)

        id2label = {
            0: "O",
            1: "separator",
        }
        label2id = {
            "O": 0,
            "separator": 1,
        }

        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
            cache_dir=model_cache_dir,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16
        )
        self.model.eval()
        self.model.to(device)

    def split_text(self, text: str) -> List[str]:
        token_chunks, offset_tup_chunks = split_text_into_even_chunks(self.tokenizer, text)
        separator_idx_lists = []
        with torch.no_grad():
            for token_chunk, offset_tup_chunk in zip(token_chunks, offset_tup_chunks):
                output = self.model(torch.tensor([token_chunk]).to("cuda"))
                logits = output.logits.cpu().numpy()
                maxes = np.max(logits, axis=-1, keepdims=True)
                shifted_exp = np.exp(logits - maxes)
                scores = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)
                token_classes = scores.argmax(axis=-1)
                # Find first occasion of each positive class (separator) in token class sequence
                separator_token_indices, = ((token_classes[0, :-1] - token_classes[0, 1:]) > 0).nonzero()

                # Consider using  [offset_tup_chunk[i][1]+1 for i in separator_token_indices]
                # Since it seems that there are several instances where a dot or semicolon is put into the "next" chunk
                separator_indices = [offset_tup_chunk[i][1] for i in separator_token_indices]
                separator_idx_lists.extend(separator_indices)
        yield from split_into_semantic_chunks(text, separator_idx_lists)

    def split_text_old(self, text: str) -> List[str]:
        text_chunks, token_chunks, _ = split_text_into_even_chunks(self.tokenizer, text)
        with torch.no_grad():
            outputs = self.pipe(text_chunks)
        for text_chunk, output in zip(text_chunks, outputs):
            yield from split_into_semantic_chunks(text_chunk, output)
