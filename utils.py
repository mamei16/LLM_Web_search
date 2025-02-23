from typing import Dict, Literal
import warnings
import math
import copy
from dataclasses import dataclass

from torch import Tensor
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, quantize_embeddings
from sentence_transformers.util import batch_to_device, truncate_embeddings


@dataclass
class Document:
    page_content: str
    metadata: Dict


class Generator:
    """Allows a generator method to return a final value after finishing
    the generation. Credit: https://stackoverflow.com/a/34073559"""
    def __init__(self, gen):
        self.gen = gen

    def __iter__(self):
        self.retval = yield from self.gen
        return self.retval


def cosine_similarity(X, Y) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices."""
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {X.shape} "
            f"and Y has shape {Y.shape}."
        )
    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    # Ignore divide by zero errors run time warnings as those are handled below.
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
    return similarity


def dict_list_to_pretty_str(data: list[dict]) -> str:
    ret_str = ""
    if isinstance(data, dict):
        data = [data]
    if isinstance(data, list):
        for i, d in enumerate(data):
            ret_str += f"Result {i+1}\n"
            ret_str += f"Title: {d['title']}\n"
            ret_str += f"{d['body']}\n"
            ret_str += f"Source URL: {d['href']}\n"
        return ret_str
    else:
        raise ValueError("Input must be dict or list[dict]")


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
    
    
class MySentenceTransformer(SentenceTransformer):
    def batch_encode(
            self,
            sentences: str | list[str],
            prompt_name: str | None = None,
            prompt: str | None = None,
            batch_size: int = 32,
            output_value: Literal["sentence_embedding", "token_embeddings"] | None = "sentence_embedding",
            precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            device: str = None,
            normalize_embeddings: bool = False,
            **kwargs,
    ) -> list[Tensor] | np.ndarray | Tensor:
        if self.device.type == "hpu" and not self.is_hpu_graph_enabled:
            import habana_frameworks.torch as ht

            ht.hpu.wrap_in_hpu_graph(self, disable_tensor_cache=True)
            self.is_hpu_graph_enabled = True

        self.eval()
        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != "sentence_embedding":
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(
                sentences, "__len__"
        ):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if prompt is None:
            if prompt_name is not None:
                try:
                    prompt = self.prompts[prompt_name]
                except KeyError:
                    raise ValueError(
                        f"Prompt name '{prompt_name}' not found in the configured prompts dictionary with keys {list(self.prompts.keys())!r}."
                    )
            elif self.default_prompt_name is not None:
                prompt = self.prompts.get(self.default_prompt_name, None)
        else:
            if prompt_name is not None:
                warnings.warn(
                    "Encode with either a `prompt`, a `prompt_name`, or neither, but not both. "
                    "Ignoring the `prompt_name` in favor of `prompt`."
                )

        extra_features = {}
        if prompt is not None:
            sentences = [prompt + sentence for sentence in sentences]

            # Some models (e.g. INSTRUCTOR, GRIT) require removing the prompt before pooling
            # Tracking the prompt length allow us to remove the prompt during pooling
            tokenized_prompt = self.tokenize([prompt])
            if "input_ids" in tokenized_prompt:
                extra_features["prompt_length"] = tokenized_prompt["input_ids"].shape[-1] - 1

        if device is None:
            device = self.device
        else:
            device = torch.device(device)

        self.to(device)

        all_embeddings = []
        tokenized_sentences = self.tokenizer(sentences, verbose=False)["input_ids"]
        batchifyer = SimilarLengthsBatchifyer(batch_size, tokenized_sentences)
        sentences = np.array(sentences)
        batch_indices = []
        for index_batch in batchifyer:
            batch_indices.append(index_batch)
            sentences_batch = sentences[index_batch]
            features = self.tokenize(sentences_batch)
            if self.device.type == "hpu":
                if "input_ids" in features:
                    curr_tokenize_len = features["input_ids"].shape
                    additional_pad_len = 2 ** math.ceil(math.log2(curr_tokenize_len[1])) - curr_tokenize_len[1]
                    features["input_ids"] = torch.cat(
                        (
                            features["input_ids"],
                            torch.ones((curr_tokenize_len[0], additional_pad_len), dtype=torch.int8),
                        ),
                        -1,
                    )
                    features["attention_mask"] = torch.cat(
                        (
                            features["attention_mask"],
                            torch.zeros((curr_tokenize_len[0], additional_pad_len), dtype=torch.int8),
                        ),
                        -1,
                    )
                    if "token_type_ids" in features:
                        features["token_type_ids"] = torch.cat(
                            (
                                features["token_type_ids"],
                                torch.zeros((curr_tokenize_len[0], additional_pad_len), dtype=torch.int8),
                            ),
                            -1,
                        )

            features = batch_to_device(features, device)
            features.update(extra_features)

            with torch.no_grad():
                out_features = self.forward(features, **kwargs)
                if self.device.type == "hpu":
                    out_features = copy.deepcopy(out_features)

                out_features["sentence_embedding"] = truncate_embeddings(
                    out_features["sentence_embedding"], self.truncate_dim
                )

                if output_value == "token_embeddings":
                    embeddings = []
                    for token_emb, attention in zip(out_features[output_value], out_features["attention_mask"]):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1

                        embeddings.append(token_emb[0: last_mask_id + 1])
                elif output_value is None:  # Return all outputs
                    embeddings = []
                    for sent_idx in range(len(out_features["sentence_embedding"])):
                        row = {name: out_features[name][sent_idx] for name in out_features}
                        embeddings.append(row)
                else:  # Sentence embeddings
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.to("cpu", non_blocking=True)
                        sync_device(device)

                all_embeddings.extend(embeddings)

        # Restore order after SimilarLengthsBatchifyer disrupted it:
        # Ensure that the order of 'indices' and 'values' matches the order of the 'texts' parameter
        batch_indices = np.concatenate(batch_indices)
        sorted_indices = np.argsort(batch_indices)
        all_embeddings = [all_embeddings[i] for i in sorted_indices]

        if precision and precision != "float32":
            all_embeddings = quantize_embeddings(all_embeddings, precision=precision)

        if convert_to_tensor:
            if len(all_embeddings):
                if isinstance(all_embeddings, np.ndarray):
                    all_embeddings = torch.from_numpy(all_embeddings)
            else:
                all_embeddings = torch.Tensor()
        elif convert_to_numpy:
            if not isinstance(all_embeddings, np.ndarray):
                if all_embeddings and all_embeddings[0].dtype == torch.bfloat16:
                    all_embeddings = np.asarray([emb.float().numpy() for emb in all_embeddings])
                else:
                    all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        elif isinstance(all_embeddings, np.ndarray):
            all_embeddings = [torch.from_numpy(embedding) for embedding in all_embeddings]

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings


def sync_device(device: torch.device):
    if device.type == "cpu":
        return
    elif device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize(device)
    else:
        warnings.warn("Device type does not match 'cuda', 'xpu' or 'mps'. Not synchronizing")
