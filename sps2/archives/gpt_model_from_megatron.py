#! /usr/bin/env python3


import math
import pickle
from contextlib import nullcontext
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from megatron import core
from megatron.core.utils import divide
from megatron.model import LayerNorm
from megatron.model.enums import AttnMaskType, AttnType, LayerType
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.fused_softmax import FusedScaleMaskSoftmax

IS_TRAINING = False
CONTEXT_LEN = 2048
HIDDEN_SIZE = 2048
FFN_HIDDEN_SIZE = 3072
NUM_HEADS = 16
VOCAB_SIZE = 50304
NUM_LAYERS = 24
EMBED_DROP_PROB = 0.1
assert HIDDEN_SIZE % NUM_HEADS == 0


class VocabEmbedding(torch.nn.Module):
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

        self.word_embeddings = VocabEmbedding()
        self.position_embeddings = torch.nn.Embedding(CONTEXT_LEN, HIDDEN_SIZE)
        self.embedding_dropout = torch.nn.Dropout(EMBED_DROP_PROB)

    def forward(self, input_ids, position_ids):
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = embeddings.transpose(0, 1).contiguous()
        # Dropout
        embeddings = self.embedding_dropout(embeddings)
        return embeddings


class QKVLinear(torch.nn.Module):
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


def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


class CoreAttention(torch.nn.Module):
    def __init__(self, layer_num, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_num = max(1, layer_num)
        self.norm_factor = math.sqrt(HIDDEN_SIZE / NUM_HEADS)
        coeff = self.layer_num
        self.norm_factor *= coeff
        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            True,
            False,
            AttnMaskType.causal,
            True,
            attention_mask_func,
            True,
            coeff,
        )
        self.attention_dropout = torch.nn.Dropout(0.1)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting input tensor: [b * np, sq, sk]
        # TODO: improve the performance
        # matmul_input_buffer = torch.empty(
        #     output_size[0] * output_size[1], output_size[2], output_size[3], dtype=query_layer.dtype
        # )
        matmul_input_buffer = torch.empty(1, device=query_layer.device)

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class AttnOutputLinear(torch.nn.Module):
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
        self.kv_channels = HIDDEN_SIZE / NUM_HEADS
        assert HIDDEN_SIZE % NUM_HEADS == 0
        self.query_key_value = QKVLinear()
        self.core_attention = CoreAttention(self.layer_num)
        self.dense = AttnOutputLinear()

    def forward(self, hidden_states, attention_mask):
        # hidden_states: [sq, b, h]

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

        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)

        output, bias = self.dense(context_layer)

        return output, bias


class MLPLinear(torch.nn.Module):
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
        self.dense_h_to_4h = MLPLinear(HIDDEN_SIZE, FFN_HIDDEN_SIZE, skip_bias_add=True)
        self.bias_gelu_fusion = True
        self.activation_func = F.gelu
        self.dense_4h_to_h = MLPLinear(FFN_HIDDEN_SIZE, HIDDEN_SIZE)

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate, bias = self.dense_h_to_4h(hidden_states)

        if self.bias_gelu_fusion:
            intermediate = bias_gelu_impl(intermediate, bias)
        else:
            if bias is not None:
                intermediate = intermediate + bias
            intermediate = self.activation_func(intermediate)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate)
        return output, output_bias


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Optional[Tensor], Tensor, float, bool) -> Tensor
    if bias is not None:
        x = x + bias
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    return out


@torch.jit.script
def bias_dropout_add_fused_train(
    x: torch.Tensor, bias: Optional[torch.Tensor], residual: torch.Tensor, prob: float
) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(
    x: torch.Tensor, bias: Optional[torch.Tensor], residual: torch.Tensor, prob: float
) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, False)


class TransformerLayer(torch.nn.Module):
    def __init__(self, layer_num, drop_path_rate, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_num = layer_num
        self.drop_path_rate = drop_path_rate
        self.input_layernorm = LayerNorm(
            HIDDEN_SIZE,
            eps=1e-5,
            no_persist_layer_norm=False,
            sequence_parallel=False,
            apply_layernorm_1p=False,
        )
        # Self attention.
        self.self_attention = Attention(layer_num)

        self.hidden_dropout = 0.1
        self.bias_dropout_fusion = True
        self.drop_path = None

        self.post_attention_layernorm = LayerNorm(
            HIDDEN_SIZE,
            eps=1e-5,
            no_persist_layer_norm=True,
            sequence_parallel=False,
            apply_layernorm_1p=False,
        )

        # MLP
        self.mlp = MLP()

        # Set bias+dropout+add fusion grad_enable execution handler.
        TORCH_MAJOR = int(torch.__version__.split(".")[0])
        TORCH_MINOR = int(torch.__version__.split(".")[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad

    def forward(self, hidden_states, attention_mask):
        layernorm_output = self.input_layernorm(hidden_states)
        attention_output, attention_bias = self.self_attention(layernorm_output, attention_mask)
        residual = hidden_states
        if IS_TRAINING:
            bias_dropout_add_func = bias_dropout_add_fused_train
        else:
            bias_dropout_add_func = bias_dropout_add_fused_inference
        if attention_bias is not None:
            attention_bias = attention_bias.expand_as(residual)
        with self.bias_dropout_add_exec_handler():
            layernorm_input = bias_dropout_add_func(attention_output, attention_bias, residual, self.hidden_dropout)
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
    def __init__(self, drop_path_rate=0.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.drop_path_rate = drop_path_rate
        self.drop_path_rates = [rate.item() for rate in torch.linspace(0, self.drop_path_rate, NUM_LAYERS)]

        def build_layer(layer_num):
            return TransformerLayer(layer_num, drop_path_rate=self.drop_path_rates[layer_num - 1])

        self.layers = []
        for i in range(NUM_LAYERS):
            layer_num = i + 1
            self.layers.append(build_layer(layer_num))
        self.layers = torch.nn.ModuleList(self.layers)

        self.final_layernorm = LayerNorm(
            HIDDEN_SIZE,
            eps=1e-5,
            no_persist_layer_norm=False,
            sequence_parallel=False,
            apply_layernorm_1p=False,
        )

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(self, hidden_states, attention_mask):
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
            hidden_states,
            requires_grad=True,
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
            hidden_states = custom(ly, ly + 1)(hidden_states, attention_mask)

        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class GPTModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = Embedding()
        self.encoder = Transformer()

        self.encoder_seq_length = CONTEXT_LEN

    def forward(self, batch):
        (enc_input_ids, enc_attn_mask, enc_position_ids, setkey_value_array, len_array) = batch
        encoder_input = self.embedding(enc_input_ids, enc_position_ids)
        encoder_output = self.encoder(encoder_input, enc_attn_mask)
        word_embeddings_weight = self.embedding.word_embeddings.weight
        logits = torch.matmul(encoder_output, word_embeddings_weight.t())
        return logits.transpose(0, 1).contiguous()


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
