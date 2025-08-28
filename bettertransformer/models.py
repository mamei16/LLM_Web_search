import logging

import torch
import torch.nn as nn
from torch.functional import F
from transformers.activations import ACT2FN

KNOWN_ACTIVATION_ATTRIBUTES = ["hidden_act", "activation", "act_fn", "activation_function"]
KNOWN_POS_EMB_ATTRIBUTES = ["position_embedding_type"]
KNOWN_NUM_LAYERS = ["num_hidden_layers", "num_layers", "encoder_layers", "n_layers"]

SUPPORTED_ACTIVATION_FUNCTIONS = ["gelu", "relu", "gelu_new"]
USE_AT_OWN_RISK_ACTIVATION_FUNCTIONS = ["quick_gelu"]


logger = logging.getLogger(__name__)

try:
    from .utils import recurse_getattr, recurse_setattr
except ImportError:
    from utils import recurse_getattr, recurse_setattr


class BetterTransformerBaseLayer:
    def __init__(
        self,
        config: "PretrainedConfig",
    ):
        r"""
        Base layer for `BetterTransformer` integration. This class is used to wrap all the necessary
        components for the `BetterTransformer` integration.

        Args:
            config (`transformers.PretrainedConfig`):
                The config of the model.
        """
        self.norm_first = False
        self.use_gelu = False
        self.act_fn = None
        self.pos_emb_type = None
        self.num_heads = None
        self.embed_dim = None
        self.num_layers = None
        self.original_layers_mapping = {}
        self.module_mapping = None
        # Some models does not have some attributes thus needs to be ignored
        # e.g. whisper does not have self_attn.k_proj.bias but has self_attn.v_proj.bias & self_attn.q_proj.bias
        self.keys_to_ignore = []

        # Get activation function
        for attr in KNOWN_ACTIVATION_ATTRIBUTES:
            if hasattr(config, attr):
                self.act_fn = getattr(config, attr)
                break

        # if act_fn not found in the config, fall back to the private `_get_activation_function` if available
        if self.act_fn is None and hasattr(self, "_get_activation_function"):
            self.act_fn = self._get_activation_function(config)

        # Get pos emb type
        for attr in KNOWN_POS_EMB_ATTRIBUTES:
            if hasattr(config, attr):
                self.pos_emb_type = getattr(config, attr)
                break

        # Get num_layers
        for attr in KNOWN_NUM_LAYERS:
            if hasattr(config, attr):
                self.num_layers = getattr(config, attr)
                break

    def validate_bettertransformer(self):
        r"""
        A wrapper function to validate the `BetterTransformer` implementation. Implements most relevant checks
        that are present in: https://github.com/pytorch/pytorch/blob/0fc7de398636f4b53e6c3fde38b4e48a5ff5b37d/torch/nn/modules/transformer.py#L457-L475
        """
        # Sanity checks
        if self.num_heads is None:
            raise ValueError("Number of heads not set for `BetterTransformer` integration.")

        if self.embed_dim is None:
            raise ValueError("Embedding dimension not set for `BetterTransformer` integration.")

        if self.norm2_eps is None or self.norm1_eps is None:
            raise ValueError("`norm2_eps` and `norm1_eps` not set for `BetterTransformer` integration.")

        # Check positional embedding
        if self.pos_emb_type is not None and self.pos_emb_type != "absolute":
            raise ValueError(
                f"Positional embedding type {self.pos_emb_type} not " "supported for `BetterTransformer` integration"
            )

        # Check norm1 epsilon and norm2 epsilon equality
        if self.norm1_eps != self.norm2_eps:
            raise ValueError("norm1_eps and norm2_eps must be equal for `BetterTransformer` integration.")

        # Check activation function
        if self.act_fn in USE_AT_OWN_RISK_ACTIVATION_FUNCTIONS:
            logger.warning(
                f"Overridding {self.act_fn} activation with gelu. Use the transformed model at your own risk, the output logits could be significantly different."
            )
            self.act_fn = "gelu"
        elif self.act_fn not in SUPPORTED_ACTIVATION_FUNCTIONS:
            raise ValueError(
                f"Activation function {self.act_fn} not supported" " for `BetterTransformer` integration."
            )
        self.use_gelu = (self.act_fn == "gelu") or (self.act_fn == "gelu_new")

        # Check num_head is even
        if self.num_heads % 2 == 1:
            raise ValueError(
                f"Number of heads {self.num_heads} is not supported"
                " for `BetterTransformer` integration."
                f" Number of heads must be even."
            )

    def _revert(self, module: torch.nn.Module) -> torch.nn.Module:
        if self.module_mapping is not None:
            if "" in self.module_mapping.values():
                for bt_module_attr_name, value in self.module_mapping.items():
                    if value == "":
                        module = getattr(self, bt_module_attr_name)
                        return module
            else:
                raise NotImplementedError("replacing a submodule in revert is not supported")

        for modified_layer_key_names, original_layer_key_names in self.original_layers_mapping.items():
            if isinstance(original_layer_key_names, list):
                current_weight = getattr(self, modified_layer_key_names)

                # Split the current weight n chunks - this is useful to split
                # the qkv layers into q, k, v layers for example.
                split_index = current_weight.shape[0] // len(original_layer_key_names)
                for i, subparam_name in enumerate(original_layer_key_names):
                    if recurse_getattr(module, subparam_name) is None:
                        # this is for example the case if bias=False is set for a nn.Linear layer
                        continue

                    if module not in self.keys_to_ignore:
                        # TODO: remove the clone once https://github.com/huggingface/transformers/pull/27314 & https://github.com/huggingface/safetensors/pull/379 are released.
                        # Safetensors is bugged when using views of tensors.
                        parameter = current_weight[i * split_index : (i + 1) * split_index].clone()
                        if isinstance(recurse_getattr(module, subparam_name), torch.nn.Parameter):
                            parameter = torch.nn.Parameter(parameter)
                        recurse_setattr(module, subparam_name, parameter)
            elif isinstance(original_layer_key_names, str):
                if recurse_getattr(module, original_layer_key_names) is None:
                    # this is for example the case if bias=False is set for a nn.Linear layer
                    continue

                parameter = getattr(self, modified_layer_key_names)
                if isinstance(recurse_getattr(module, original_layer_key_names), torch.nn.Parameter):
                    parameter = torch.nn.Parameter(parameter)
                recurse_setattr(module, original_layer_key_names, parameter)
            else:
                raise ValueError(
                    f"Invalid type {type(modified_layer_key_names)} for `original_layers_mapping`",
                    " please use either `str` or `list`.",
                )
        return module

class BertLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):
    def __init__(self, bert_layer, config):
        r"""
        A simple conversion of the BERT layer to its `BetterTransformer` implementation.

        Args:
            bert_layer (`torch.nn.Module`):
                The original BERT Layer where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        # In_proj layer
        self.in_proj_weight = nn.Parameter(
            torch.cat(
                [
                    bert_layer.attention.self.query.weight,
                    bert_layer.attention.self.key.weight,
                    bert_layer.attention.self.value.weight,
                ]
            )
        )
        self.in_proj_bias = nn.Parameter(
            torch.cat(
                [
                    bert_layer.attention.self.query.bias,
                    bert_layer.attention.self.key.bias,
                    bert_layer.attention.self.value.bias,
                ]
            )
        )

        # Out proj layer
        self.out_proj_weight = bert_layer.attention.output.dense.weight
        self.out_proj_bias = bert_layer.attention.output.dense.bias

        # Linear layer 1
        self.linear1_weight = bert_layer.intermediate.dense.weight
        self.linear1_bias = bert_layer.intermediate.dense.bias

        # Linear layer 2
        self.linear2_weight = bert_layer.output.dense.weight
        self.linear2_bias = bert_layer.output.dense.bias

        # Layer norm 1
        self.norm1_eps = bert_layer.attention.output.LayerNorm.eps
        self.norm1_weight = bert_layer.attention.output.LayerNorm.weight
        self.norm1_bias = bert_layer.attention.output.LayerNorm.bias

        # Layer norm 2
        self.norm2_eps = bert_layer.output.LayerNorm.eps
        self.norm2_weight = bert_layer.output.LayerNorm.weight
        self.norm2_bias = bert_layer.output.LayerNorm.bias

        # Model hyper parameters
        self.num_heads = bert_layer.attention.self.num_attention_heads
        self.embed_dim = bert_layer.attention.self.all_head_size

        # Last step: set the last layer to `False` -> this will be set to `True` when converting the model
        self.is_last_layer = False

        self.original_layers_mapping = {
            "in_proj_weight": [
                "attention.self.query.weight",
                "attention.self.key.weight",
                "attention.self.value.weight",
            ],
            "in_proj_bias": ["attention.self.query.bias", "attention.self.key.bias", "attention.self.value.bias"],
            "out_proj_weight": "attention.output.dense.weight",
            "out_proj_bias": "attention.output.dense.bias",
            "linear1_weight": "intermediate.dense.weight",
            "linear1_bias": "intermediate.dense.bias",
            "linear2_weight": "output.dense.weight",
            "linear2_bias": "output.dense.bias",
            "norm1_eps": "attention.output.LayerNorm.eps",
            "norm1_weight": "attention.output.LayerNorm.weight",
            "norm1_bias": "attention.output.LayerNorm.bias",
            "norm2_eps": "output.LayerNorm.eps",
            "norm2_weight": "output.LayerNorm.weight",
            "norm2_bias": "output.LayerNorm.bias",
        }
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.act_fn_callable = ACT2FN[self.act_fn]

        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, *_, **kwargs):
        # No check on output_attentions here as roformer relies on BertLayerBetterTransformer but does not pass output_attentions as keyword argument.
        if not self.training and not torch._C._is_any_autocast_enabled():
            if hidden_states.is_nested:
                attention_mask = None

            if attention_mask is not None:
                # attention mask comes in with values 0 and -inf. we convert to torch.nn.TransformerEncoder style bool mask
                # 0->false->keep this token -inf->true->mask this token
                attention_mask = attention_mask.bool()
                attention_mask = torch.reshape(attention_mask, (attention_mask.shape[0], attention_mask.shape[-1]))
                hidden_states = torch._nested_tensor_from_mask(hidden_states, ~attention_mask)
                attention_mask = None

            hidden_states = torch._transformer_encoder_layer_fwd(
                hidden_states,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj_weight,
                self.out_proj_bias,
                self.use_gelu,
                self.norm_first,
                self.norm1_eps,
                self.norm1_weight,
                self.norm1_bias,
                self.norm2_weight,
                self.norm2_bias,
                self.linear1_weight,
                self.linear1_bias,
                self.linear2_weight,
                self.linear2_bias,
                attention_mask,
            )
            if hidden_states.is_nested and self.is_last_layer:
                hidden_states = hidden_states.to_padded_tensor(0.0)
        else:
            qkv = F.linear(hidden_states, weight=self.in_proj_weight, bias=self.in_proj_bias)

            qkv = qkv.view(qkv.size()[:-1] + (3, self.num_heads, self.attention_head_size)).permute(2, 0, 3, 1, 4)
            query, key, value = qkv[0], qkv[1], qkv[2]

            # NOTE: In PyTorch 2.0, passing an attention_mask will automatically dispatch
            # to the "math" path and will NOT use flash attention / memory-efficient attention.
            # We should support xformers / Hazy-flash / rocm-flash directly and stop relying on PyTorch to do the work.
            if self.training:
                attention_mask = None
            attention_out = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                is_causal=False,
                dropout_p=self.attention_probs_dropout_prob if self.training else 0.0,
            )

            attention_out = attention_out.permute(0, 2, 1, 3).contiguous()
            new_attention_out_shape = attention_out.size()[:-2] + (self.num_heads * self.attention_head_size,)
            attention_out = attention_out.view(new_attention_out_shape)

            # BertSelfOutput
            attention_out = F.layer_norm(
                F.dropout(
                    F.linear(attention_out, self.out_proj_weight, self.out_proj_bias),
                    p=self.hidden_dropout_prob,
                    training=self.training,
                )
                + hidden_states,
                normalized_shape=self.norm1_weight.shape,
                weight=self.norm1_weight,
                bias=self.norm1_bias,
            )

            # BertIntermediate
            hidden_states = self.act_fn_callable(F.linear(attention_out, self.linear1_weight, self.linear1_bias))

            # BertOutput
            hidden_states = F.layer_norm(
                attention_out
                + F.dropout(
                    F.linear(hidden_states, self.linear2_weight, self.linear2_bias),
                    p=self.hidden_dropout_prob,
                    training=self.training,
                ),
                normalized_shape=self.norm2_weight.shape,
                weight=self.norm2_weight,
                bias=self.norm2_bias,
            )

        return (hidden_states,)


class DistilBertLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):
    def __init__(self, bert_layer, config):
        r"""
        A simple conversion of the Distill-BERTLayer to its `BetterTransformer` implementation.

        Args:
            bert_layer (`torch.nn.Module`):
                The original Distill-BERT Layer where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        # In_proj layer
        self.in_proj_weight = nn.Parameter(
            torch.cat(
                [
                    bert_layer.attention.q_lin.weight,
                    bert_layer.attention.k_lin.weight,
                    bert_layer.attention.v_lin.weight,
                ]
            )
        )
        self.in_proj_bias = nn.Parameter(
            torch.cat(
                [
                    bert_layer.attention.q_lin.bias,
                    bert_layer.attention.k_lin.bias,
                    bert_layer.attention.v_lin.bias,
                ]
            )
        )

        # Out proj layer
        self.out_proj_weight = bert_layer.attention.out_lin.weight
        self.out_proj_bias = bert_layer.attention.out_lin.bias

        # Linear layer 1
        self.linear1_weight = bert_layer.ffn.lin1.weight
        self.linear1_bias = bert_layer.ffn.lin1.bias

        # Linear layer 2
        self.linear2_weight = bert_layer.ffn.lin2.weight
        self.linear2_bias = bert_layer.ffn.lin2.bias

        # Layer norm 1
        self.norm1_eps = bert_layer.sa_layer_norm.eps
        self.norm1_weight = bert_layer.sa_layer_norm.weight
        self.norm1_bias = bert_layer.sa_layer_norm.bias

        # Layer norm 2
        self.norm2_eps = bert_layer.output_layer_norm.eps
        self.norm2_weight = bert_layer.output_layer_norm.weight
        self.norm2_bias = bert_layer.output_layer_norm.bias

        # Model hyper parameters
        self.num_heads = bert_layer.attention.n_heads
        self.embed_dim = bert_layer.attention.dim

        # Last step: set the last layer to `False` -> this will be set to `True` when converting the model
        self.is_last_layer = False

        self.original_layers_mapping = {
            "in_proj_weight": ["attention.q_lin.weight", "attention.k_lin.weight", "attention.v_lin.weight"],
            "in_proj_bias": ["attention.q_lin.bias", "attention.k_lin.bias", "attention.v_lin.bias"],
            "out_proj_weight": "attention.out_lin.weight",
            "out_proj_bias": "attention.out_lin.bias",
            "linear1_weight": "ffn.lin1.weight",
            "linear1_bias": "ffn.lin1.bias",
            "linear2_weight": "ffn.lin2.weight",
            "linear2_bias": "ffn.lin2.bias",
            "norm1_weight": "sa_layer_norm.weight",
            "norm1_bias": "sa_layer_norm.bias",
            "norm2_weight": "output_layer_norm.weight",
            "norm2_bias": "output_layer_norm.bias",
        }
        self.attention_dropout = config.attention_dropout
        self.dropout = config.dropout
        self.attention_head_size = config.dim // config.n_heads
        self.act_fn_callable = ACT2FN[self.act_fn]

        self.validate_bettertransformer()

    def forward(self, hidden_states, attn_mask, output_attentions: bool, head_mask=None, encoder_attention_mask=None,
                *_):
        if output_attentions:
            raise ValueError("output_attentions=True can not be supported with BetterTransformer.")

        if not self.training and not torch.is_autocast_enabled() and not torch.is_autocast_cpu_enabled():
            if hidden_states.is_nested:
                attn_mask = None

            if attn_mask is not None:
                # attention mask comes in with values 0 and -inf. we convert to torch.nn.TransformerEncoder style bool mask
                # 0->false->keep this token -inf->true->mask this token
                attn_mask = attn_mask.bool()
                attn_mask = torch.reshape(attn_mask, (attn_mask.shape[0], attn_mask.shape[-1]))
                seqlen = attn_mask.shape[1]
                lengths = torch.sum(~attn_mask, 1)
                if not all(l == seqlen for l in lengths):
                    hidden_states = torch._nested_tensor_from_mask(hidden_states, attn_mask)
                attn_mask = None

            hidden_states = torch._transformer_encoder_layer_fwd(
                hidden_states,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj_weight,
                self.out_proj_bias,
                self.use_gelu,
                self.norm_first,
                self.norm1_eps,
                self.norm1_weight,
                self.norm1_bias,
                self.norm2_weight,
                self.norm2_bias,
                self.linear1_weight,
                self.linear1_bias,
                self.linear2_weight,
                self.linear2_bias,
                attn_mask,
            )
            if hidden_states.is_nested and self.is_last_layer:
                hidden_states = hidden_states.to_padded_tensor(0.0)
        else:
            qkv = F.linear(hidden_states, weight=self.in_proj_weight, bias=self.in_proj_bias)

            qkv = qkv.view(qkv.size()[:-1] + (3, self.num_heads, self.attention_head_size)).permute(2, 0, 3, 1, 4)
            query, key, value = qkv[0], qkv[1], qkv[2]

            # TODO: Kind of stupid to do that at each layer, should be fixed in transformers
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2).to(dtype=query.dtype)
            attn_mask = (1.0 - attn_mask) * torch.finfo(query.dtype).min

            # NOTE: In PyTorch 2.0, passing an attention_mask will automatically dispatch
            # to the "math" path and will NOT use flash attention / memory-efficient attention.
            # We should support xformers / Hazy-flash / rocm-flash directly and stop relying on PyTorch to do the work.
            if self.training:
                attn_mask = None
            attention_out = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                is_causal=False,
                dropout_p=self.attention_dropout if self.training else 0.0,
            )

            attention_out = attention_out.permute(0, 2, 1, 3).contiguous()
            new_attention_out_shape = attention_out.size()[:-2] + (self.num_heads * self.attention_head_size,)
            attention_out = attention_out.view(new_attention_out_shape)

            # BertSelfOutput
            attention_out = F.layer_norm(
                F.dropout(
                    F.linear(attention_out, self.out_proj_weight, self.out_proj_bias),
                    p=self.dropout,
                    training=self.training,
                )
                + hidden_states,
                normalized_shape=self.norm1_weight.shape,
                weight=self.norm1_weight,
                bias=self.norm1_bias,
            )

            # BertIntermediate
            hidden_states = self.act_fn_callable(F.linear(attention_out, self.linear1_weight, self.linear1_bias))

            # BertOutput
            hidden_states = F.layer_norm(
                attention_out
                + F.dropout(
                    F.linear(hidden_states, self.linear2_weight, self.linear2_bias),
                    p=self.dropout,
                    training=self.training,
                ),
                normalized_shape=self.norm2_weight.shape,
                weight=self.norm2_weight,
                bias=self.norm2_bias,
            )
        return (hidden_states,)

class BetterTransformerManager:
    MODEL_MAPPING = {
        "bert": {"BertLayer": BertLayerBetterTransformer},
        "distilbert": {"TransformerBlock": DistilBertLayerBetterTransformer},
    }

    OVERWRITE_METHODS = {
        # "llama": {"LlamaModel": ("_prepare_decoder_attention_mask", _llama_prepare_decoder_attention_mask)}
    }

    EXCLUDE_FROM_TRANSFORM = {
        # clip's text model uses causal attention, that is most likely not supported in BetterTransformer
        "clip": ["text_model"],
        # blip-2's Q-former and vision model should not be identified as the last layers of the model
        "blip-2": ["qformer.encoder.layer", "vision_model.encoder.layers"],
        # bark.codec_model.encoder is not supported in BetterTransformer
        "bark": ["codec_model.encoder.layers"],
    }

    CAN_NOT_BE_SUPPORTED = {
        "deberta-v2": "DeBERTa v2 does not use a regular attention mechanism, which is not supported in PyTorch's BetterTransformer.",
        "glpn": "GLPN has a convolutional layer present in the FFN network, which is not supported in PyTorch's BetterTransformer.",
    }

    NOT_REQUIRES_NESTED_TENSOR = {
        "bark",
        "blenderbot",
        "bloom",
        "codegen",
        "gpt2",
        "gptj",
        "gpt_neo",
        "gpt_neox",
        "opt",
        "pegasus",
        "t5",
    }

    NOT_REQUIRES_STRICT_VALIDATION = {
        "blenderbot",
        "blip-2",
        "bloom",
        "codegen",
        "gpt2",
        "gptj",
        "gpt_neo",
        "gpt_neox",
        "opt",
        "pegasus",
        "t5",
    }

    @staticmethod
    def cannot_support(model_type: str) -> bool:
        """
        Returns True if a given model type can not be supported by PyTorch's Better Transformer.

        Args:
            model_type (`str`):
                The model type to check.
        """
        return model_type in BetterTransformerManager.CAN_NOT_BE_SUPPORTED

    @staticmethod
    def supports(model_type: str) -> bool:
        """
        Returns True if a given model type is supported by PyTorch's Better Transformer, and integrated in Optimum.

        Args:
            model_type (`str`):
                The model type to check.
        """
        return model_type in BetterTransformerManager.MODEL_MAPPING

    @staticmethod
    def requires_nested_tensor(model_type: str) -> bool:
        """
        Returns True if the BetterTransformer implementation for a given architecture uses nested tensors, False otherwise.

        Args:
            model_type (`str`):
                The model type to check.
        """
        return model_type not in BetterTransformerManager.NOT_REQUIRES_NESTED_TENSOR

    @staticmethod
    def requires_strict_validation(model_type: str) -> bool:
        """
        Returns True if the architecture requires to make sure all conditions of `validate_bettertransformer` are met.

        Args:
            model_type (`str`):
                The model type to check.
        """
        return model_type not in BetterTransformerManager.NOT_REQUIRES_STRICT_VALIDATION