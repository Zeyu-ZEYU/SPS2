#! /usr/bin/env python3


import pickle
from contextlib import nullcontext
from typing import List, Optional

import einops
import torch
import torch.nn.functional as F
from apex.normalization.fused_layer_norm import MixedFusedLayerNorm
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from torch.nn.parameter import Parameter

from megatron import core
from megatron.core.utils import divide

# from megatron.model import LayerNorm
# from megatron.model.enums import AttnMaskType
from megatron.model.fused_bias_gelu import bias_gelu_impl

# from megatron.model.fused_softmax import FusedScaleMaskSoftmax

CONTEXT_LEN = 2048
HIDDEN_SIZE = 2048
NUM_LAYERS = 24
NUM_HEADS = 16
MLP_HIDDEN_SIZE = 3072
VOCAB_SIZE = 50304
assert HIDDEN_SIZE % NUM_HEADS == 0


class _VocabEmbedding(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weight = Parameter(
            torch.empty(
                VOCAB_SIZE,
                HIDDEN_SIZE,
                dtype=torch.float16,
            )
        )

    def forward(self, input_):
        masked_input = input_
        output = F.embedding(masked_input, self.weight, None, None, 2.0, False, False)
        return output


class Embedding(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.word_embeddings = _VocabEmbedding()
        self.position_embeddings = torch.nn.Embedding(CONTEXT_LEN, HIDDEN_SIZE, dtype=torch.float16)
        self.embedding_dropout = torch.nn.Dropout(0.1)

    def forward(self, input_ids, position_ids):
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = embeddings.transpose(0, 1).contiguous()
        # Dropout
        embeddings = self.embedding_dropout(embeddings)
        return embeddings


class _QKVLinear(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weight = Parameter(
            torch.empty(
                3 * HIDDEN_SIZE,
                HIDDEN_SIZE,
                dtype=torch.float16,
            )
        )
        self.bias = Parameter(torch.empty(3 * HIDDEN_SIZE, dtype=torch.float16))

    def forward(self, input_):
        output = torch.matmul(input_, self.weight.t())
        return output + self.bias, None


class _CoreAttention(torch.nn.Module):
    def __init__(self, layer_num, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_num = layer_num

    def forward(self, query_layer, key_layer, value_layer):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================
        batch_size = query_layer.size(1)
        seqlen_q = query_layer.size(0)
        seqlen_k = key_layer.size(0)
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=query_layer.device)
        cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=query_layer.device)
        query_layer = einops.rearrange(query_layer, "s b h d -> (s b) h d")
        key_layer = einops.rearrange(key_layer, "s b h d -> (s b) h d")
        value_layer = einops.rearrange(value_layer, "s b h d -> (s b) h d")
        output = flash_attn_varlen_func(
            query_layer,
            key_layer,
            value_layer,
            cu_seqlens_q,
            cu_seqlens_k,
            seqlen_q,
            seqlen_k,
            0.1,
            causal=True,
        )
        output = einops.rearrange(output, "(s b) h d -> s b (h d)", b=batch_size, s=seqlen_q)
        return output


class _PostAttnLinear(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weight = Parameter(
            torch.empty(
                HIDDEN_SIZE,
                HIDDEN_SIZE,
                dtype=torch.float16,
            )
        )
        self.bias = Parameter(torch.empty(HIDDEN_SIZE, dtype=torch.float16))

    def forward(self, input_):
        output = torch.matmul(input_, self.weight.t())
        return output, self.bias


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class Attention(torch.nn.Module):
    def __init__(self, layer_num, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_num = layer_num
        self.kv_channels = HIDDEN_SIZE // NUM_HEADS
        self.query_key_value = _QKVLinear()
        self.core_attention = _CoreAttention(self.layer_num)
        self.dense = _PostAttnLinear()

        # Inference key-value memory
        self.inference_key_memory = None
        self.inference_value_memory = None
        self.inference_current_sequence_len = 0

    def forward(self, hidden_states, set_inference_key_value_memory=False, inference_max_sequence_len=None):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        if set_inference_key_value_memory:
            assert inference_max_sequence_len and inference_max_sequence_len > 0
            self.inference_key_memory = self._allocate_memory(
                inference_max_sequence_len, hidden_states.size(1), hidden_states.dtype, hidden_states.device
            )
            self.inference_value_memory = self._allocate_memory(
                inference_max_sequence_len, hidden_states.size(1), hidden_states.dtype, hidden_states.device
            )
            self.inference_current_sequence_len = 0

        # Some consistency check.
        if inference_max_sequence_len:
            assert self.inference_current_sequence_len < self.inference_key_memory.size(0)
            assert inference_max_sequence_len == self.inference_key_memory.size(0)
        # This is added for safety. In case inference_max_sequence_len
        # is not provided, make sure there is no potential memory left
        # from previous inference.
        if not inference_max_sequence_len:
            self.inference_key_memory = None
            self.inference_value_memory = None

        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            NUM_HEADS,
            3 * HIDDEN_SIZE // NUM_HEADS,
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # ===================================================
        # Adjust key, value, and attention mask for inference
        # ===================================================
        if inference_max_sequence_len:
            # Adjust the range variables.
            start = self.inference_current_sequence_len
            self.inference_current_sequence_len += key_layer.size(0)
            end = self.inference_current_sequence_len
            # Copy key and values.
            self.inference_key_memory[start:end, ...] = key_layer
            self.inference_value_memory[start:end, ...] = value_layer
            key_layer = self.inference_key_memory[:end, ...]
            value_layer = self.inference_value_memory[:end, ...]

        context_layer = self.core_attention(query_layer, key_layer, value_layer)

        output, bias = self.dense(context_layer)

        return output, bias

    def _allocate_memory(self, inference_max_sequence_len, batch_size, dtype, device):
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            NUM_HEADS,
            HIDDEN_SIZE // NUM_HEADS,
            dtype=dtype,
            device=device,
        )


class _MLPLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, skip_bias_add=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.skip_bias_add = skip_bias_add
        self.weight = Parameter(
            torch.empty(
                output_size,
                input_size,
                dtype=torch.float16,
            )
        )
        self.bias = Parameter(torch.empty(output_size, dtype=torch.float16))

    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None
        output = torch.matmul(input_, self.weight.t())
        if bias is not None:
            output += bias
            output_bias = None
        else:
            output_bias = self.bias
        return output, output_bias


class MLP(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dense_h_to_xh = _MLPLinear(HIDDEN_SIZE, MLP_HIDDEN_SIZE, skip_bias_add=True)
        self.bias_gelu_fusion = True
        self.activation_func = F.gelu
        self.dense_xh_to_h = _MLPLinear(MLP_HIDDEN_SIZE, HIDDEN_SIZE)

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate, bias = self.dense_h_to_xh(hidden_states)

        if self.bias_gelu_fusion:
            intermediate = bias_gelu_impl(intermediate, bias)
        else:
            if bias is not None:
                intermediate = intermediate + bias
            intermediate = self.activation_func(intermediate)

        # [s, b, h]
        output, output_bias = self.dense_xh_to_h(intermediate)
        return output, output_bias


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Optional[Tensor], Tensor, float, bool) -> Tensor
    if bias is not None:
        x = x + bias
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    return out


@torch.jit.script
def bias_dropout_add_fused_inference(
    x: torch.Tensor, bias: Optional[torch.Tensor], residual: torch.Tensor, prob: float
) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, False)


class TransformerLayer(torch.nn.Module):
    def __init__(self, layer_num, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_num = layer_num
        self.input_layernorm = MixedFusedLayerNorm(HIDDEN_SIZE, 1e-5, sequence_parallel_enbaled=False)
        # Self attention.
        self.self_attention = Attention(layer_num)

        self.hidden_dropout = 0.1

        self.post_attention_layernorm = MixedFusedLayerNorm(HIDDEN_SIZE, 1e-5, sequence_parallel_enbaled=False)

        # MLP
        self.mlp = MLP()

        # Set bias+dropout+add fusion grad_enable execution handler.
        TORCH_MAJOR = int(torch.__version__.split(".")[0])
        TORCH_MINOR = int(torch.__version__.split(".")[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad

    def forward(self, hidden_states, set_inference_key_value_memory=False, inference_max_sequence_len=None):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            layernorm_output = self.input_layernorm(hidden_states)
        attention_output, attention_bias = self.self_attention(
            layernorm_output, set_inference_key_value_memory, inference_max_sequence_len
        )
        residual = hidden_states
        bias_dropout_add_func = bias_dropout_add_fused_inference
        if attention_bias is not None:
            attention_bias = attention_bias.expand_as(residual)
        with self.bias_dropout_add_exec_handler():
            layernorm_input = bias_dropout_add_func(attention_output, attention_bias, residual, self.hidden_dropout)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_bias = torch.tensor(0.0, device=layernorm_output.device, dtype=layernorm_output.dtype)

        mlp_output, mlp_bias = self.mlp(layernorm_output)

        residual = layernorm_input

        if mlp_bias is not None:
            mlp_bias = mlp_bias.expand_as(residual)
        with self.bias_dropout_add_exec_handler():
            output = bias_dropout_add_func(mlp_output, mlp_bias, residual, self.hidden_dropout)
        output = core.utils.make_viewless_tensor(inp=output, requires_grad=output.requires_grad, keep_graph=True)

        return output


class Transformer(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        def _build_layer(layer_num):
            return TransformerLayer(layer_num)

        self.layers = []
        for i in range(NUM_LAYERS):
            layer_num = i + 1
            self.layers.append(_build_layer(layer_num))
        self.layers = torch.nn.ModuleList(self.layers)

        self.final_layernorm = MixedFusedLayerNorm(HIDDEN_SIZE, 1e-5, sequence_parallel_enbaled=False)

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(self, hidden_states, set_inference_key_value_memory=False, inference_max_sequence_len=None):
        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = core.utils.make_viewless_tensor(
            inp=hidden_states,
            requires_grad=hidden_states.requires_grad,
            keep_graph=True,
        )

        def custom(start, end):
            def custom_forward(*args, **kwargs):
                x_, *args = args
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, *args, **kwargs)
                return x_

            return custom_forward

        for ly in range(NUM_LAYERS):
            hidden_states = custom(ly, ly + 1)(hidden_states, set_inference_key_value_memory, inference_max_sequence_len)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class GPTTokenizer:
    def __init__(self, tokenizer_pickle_path) -> None:
        self.tokenizer = None
        with open(tokenizer_pickle_path, "rb") as file:
            self.tokenizer = pickle.load(file)

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    def text_to_tokens(self, text):
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def tokens_to_text(self, tokens):
        text = self.tokenizer.convert_tokens_to_string(tokens)
        return text

    def token_to_id(self, token):
        return self.tokens_to_ids([token])[0]

    def tokens_to_ids(self, tokens):
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids

    def ids_to_tokens(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return tokens

    def text_to_ids(self, text):
        tokens = self.text_to_tokens(text)
        ids = self.tokens_to_ids(tokens)
        return ids

    def ids_to_text(self, ids):
        tokens = self.ids_to_tokens(ids)
        tokens_clean = [t for t in tokens if t not in self.tokenizer.all_special_tokens]
        text = self.tokens_to_text(tokens_clean)
        return text

    @property
    def vocab(self):
        id2vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        return [id2vocab[i] for i in range(len(id2vocab))]

    @property
    def pad_id(self):
        return self.tokens_to_ids([None])[0]

    @property
    def bos_id(self):
        return self.tokens_to_ids(["<|endoftext|>"])[0]

    @property
    def eos_id(self):
        return self.tokens_to_ids(["<|endoftext|>"])[0]

    @property
    def sep_id(self):
        return self.tokens_to_ids(["<|endoftext|>"])[0]

    @property
    def cls_id(self):
        return self.tokens_to_ids(["<|endoftext|>"])[0]

    @property
    def unk_id(self):
        return self.tokens_to_ids(["<|endoftext|>"])[0]

    @property
    def mask_id(self):
        return self.tokens_to_ids([None])[0]

    @property
    def name(self):
        return type(self.tokenizer).__name__


class GPTModel(torch.nn.Module):
    def __init__(self, param_pickle_path=None, device: str = "cuda", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.enc_total_context_len = CONTEXT_LEN

        self.device = torch.device(device)

        torch.cuda.set_device(self.device)

        self.embedding = Embedding()
        self.encoder = Transformer()

        if param_pickle_path:
            self._init_model(param_pickle_path)

        self.to(self.device)
        self.eval()

    def forward(self, inputs):
        (enc_input_ids, enc_position_ids, set_inference_key_value_memory, inference_max_sequence_len) = inputs
        enc_input_ids = enc_input_ids.to(self.device)
        enc_position_ids = enc_position_ids.to(self.device)
        encoder_input = self.embedding(enc_input_ids, enc_position_ids)
        encoder_output = self.encoder(encoder_input, set_inference_key_value_memory, inference_max_sequence_len)
        word_embeddings_weight = self.embedding.word_embeddings.weight
        logits = torch.matmul(encoder_output, word_embeddings_weight.t())
        return logits.transpose(0, 1).contiguous()

    def _init_model(self, param_pickle_path):
        with open(param_pickle_path, "rb") as file:
            params = pickle.load(file)
        for (key, _), (_, param) in zip(self.named_parameters(), params):
            self.state_dict()[key].copy_(param.half())
