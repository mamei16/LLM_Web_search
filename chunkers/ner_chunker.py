from typing import List
from attr import dataclass

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
try:
    from ..chunkers.base_chunker import TextSplitter
except:
    from chunkers.base_chunker import TextSplitter


def split_into_chunks(lst, n):
    last_item_shorter = False
    if len(lst[-1]) < len(lst[0]):
        last_item_shorter = True
        max_index = len(lst)-1
    else:
        max_index = len(lst)

    for i in range(0, max_index, n):
        yield lst[i : min(i + n, max_index)]

    if last_item_shorter:
        yield lst[-1:]


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


@dataclass
class Token:
    index: int
    start: int
    end: int
    length: int
    decoded_str: str  #TODO: for debugging, remove


class NerChunker(TextSplitter):
    def __init__(self, model_name="mirth/chonky_distilbert_base_uncased_1", device="cpu", model_cache_dir: str = None):
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
        max_seq_len = self.tokenizer.model_max_length
        window_step_size = max_seq_len // 2
        ids_plus = self.tokenizer(text, truncation=True, add_special_tokens=True, return_offsets_mapping=True,
                                  return_overflowing_tokens=True, stride=window_step_size)

        tokens = [[Token(i*max_seq_len+j,
                         offset_tup[0], offset_tup[1],
                         offset_tup[1]-offset_tup[0],
                         text[offset_tup[0]:offset_tup[1]]) for j, offset_tup in enumerate(offset_list)]
                  for i, offset_list in enumerate(ids_plus["offset_mapping"])]

        input_ids = ids_plus["input_ids"]
        all_separator_tokens = []

        batch_size = 4
        for input_id_chunk, token_chunk in zip(split_into_chunks(input_ids, batch_size),
                                                 split_into_chunks(tokens, batch_size)):
            with torch.no_grad():
                output = self.model(torch.tensor(input_id_chunk).to("cuda"))

            logits = output.logits.cpu().numpy()
            maxes = np.max(logits, axis=-1, keepdims=True)
            shifted_exp = np.exp(logits - maxes)
            scores = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)
            token_classes = scores.argmax(axis=-1)
            # Find last index of each sequence of ones in token class sequence
            separator_token_idx_tup = ((token_classes[:, :-1] - token_classes[:, 1:]) > 0).nonzero()

            separator_tokens = [token_chunk[i][j] for i, j in zip(*separator_token_idx_tup)]
            all_separator_tokens.extend(separator_tokens)


        flat_tokens = [token for window in tokens for token in window]
        sorted_separator_tokens = sorted(all_separator_tokens, key=lambda x: x.start)
        separator_indices = []
        for i in range(len(sorted_separator_tokens)-1):
            current_token = sorted_separator_tokens[i]
            next_sep_token = sorted_separator_tokens[i+1]

            while current_token.end == flat_tokens[current_token.index+1].start:
                # TODO: Maybe avoid crossing certain symbols here
                #  like dots (check that a dot is not followed by str.isalpha()), or newlines(?)
                current_token = flat_tokens[current_token.index+1]

            if ((current_token.end == 0) or
                (current_token.start + current_token.length) > next_sep_token.start or
                ((next_sep_token.end - current_token.end) <= 1)):
                continue

            separator_indices.append(current_token.end)

        separator_indices.append(sorted_separator_tokens[-1].end)

        yield from split_into_semantic_chunks(text, separator_indices)
