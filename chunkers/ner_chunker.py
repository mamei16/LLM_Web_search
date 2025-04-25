from typing import List
from attr import dataclass

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
try:
    from ..chunkers.base_chunker import TextSplitter
    from ..chunkers.character_chunker import RecursiveCharacterTextSplitter
except:
    from chunkers.base_chunker import TextSplitter
    from chunkers.character_chunker import RecursiveCharacterTextSplitter


def batchify(lst, batch_size):
    last_item_shorter = False
    if len(lst[-1]) < len(lst[0]):
        last_item_shorter = True
        max_index = len(lst)-1
    else:
        max_index = len(lst)

    for i in range(0, max_index, batch_size):
        yield lst[i : min(i + batch_size, max_index)]

    if last_item_shorter:
        yield lst[-1:]


@dataclass
class Token:
    index: int
    start: int
    end: int
    length: int
    decoded_str: str


class TokenClassificationChunker(TextSplitter):
    def __init__(self, model_id="mirth/chonky_distilbert_base_uncased_1", device="cpu", model_cache_dir: str = None,
                 max_chunk_size: int = 99999):
        super().__init__()
        self.device = device
        self.is_modernbert = model_id == "mirth/chonky_modernbert_base_1"
        self.max_chunk_size = max_chunk_size
        self.character_splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size, chunk_overlap=10,
                                                                 separators=["\n\n", "\n", ".", ", ", " ", ""])
        id2label = {
            0: "O",
            1: "separator",
        }
        label2id = {
            "O": 0,
            "separator": 1,
        }

        if self.is_modernbert:
            tokenizer_kwargs = {"model_max_length": 1024}
        else:
            tokenizer_kwargs = {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=model_cache_dir, **tokenizer_kwargs)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_id,
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
            cache_dir=model_cache_dir,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16
        )
        self.model.eval()
        self.model.to(device)

    def split_into_semantic_chunks(self, text, separator_indices: List[int]):
        start_index = 0

        for idx in separator_indices:
            chunk = text[start_index:idx].strip()
            if len(chunk) > self.max_chunk_size:
                yield from self.character_splitter.split_text(chunk)
            else:
                yield chunk
            start_index = idx

        if start_index < len(text):
            yield text[start_index:].strip()

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
        for input_id_batch, token_batch in zip(batchify(input_ids, batch_size),
                                               batchify(tokens, batch_size)):
            with torch.no_grad():
                output = self.model(torch.tensor(input_id_batch).to(self.device))

            logits = output.logits.cpu().numpy()
            maxes = np.max(logits, axis=-1, keepdims=True)
            shifted_exp = np.exp(logits - maxes)
            scores = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)
            token_classes = scores.argmax(axis=-1)
            # Find last index of each sequence of ones in token class sequence
            separator_token_idx_tup = ((token_classes[:, :-1] - token_classes[:, 1:]) > 0).nonzero()

            separator_tokens = [token_batch[i][j] for i, j in zip(*separator_token_idx_tup)]
            all_separator_tokens.extend(separator_tokens)

        flat_tokens = [token for window in tokens for token in window]
        sorted_separator_tokens = sorted(all_separator_tokens, key=lambda x: x.start)
        separator_indices = []
        for i in range(len(sorted_separator_tokens)-1):
            current_sep_token = sorted_separator_tokens[i]
            if current_sep_token.end == 0:
                continue
            next_sep_token = sorted_separator_tokens[i+1]
            # next_token is the token succeeding current_sep_token in the original text
            next_token = flat_tokens[current_sep_token.index+1]

            # If current separator token is part of a bigger contiguous token, move to the end of the bigger token
            while (current_sep_token.end == next_token.start and
                   (not self.is_modernbert or (current_sep_token.decoded_str != '\n'
                                               and not next_token.decoded_str.startswith(' ')))):
                current_sep_token = next_token
                next_token = flat_tokens[current_sep_token.index+1]

            if ((current_sep_token.start + current_sep_token.length) > next_sep_token.start or
                ((next_sep_token.end - current_sep_token.end) <= 1)):
                continue

            separator_indices.append(current_sep_token.end)

        if sorted_separator_tokens:
            separator_indices.append(sorted_separator_tokens[-1].end)

        yield from self.split_into_semantic_chunks(text, separator_indices)
