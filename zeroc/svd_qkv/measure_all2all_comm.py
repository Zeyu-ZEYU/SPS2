#! /usr/bin/env python3


import math
import pickle
import time
from contextlib import nullcontext
from typing import Any, List, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from apex.normalization.fused_layer_norm import MixedFusedLayerNorm
from torch.nn.parameter import Parameter

from megatron import core
from megatron.core.utils import divide

# from megatron.model import LayerNorm
from megatron.model.enums import AttnMaskType
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.fused_softmax import FusedScaleMaskSoftmax

# These hyper-parameters are noly for GPTModel rather than DistributedGPTModel
IS_TRAINING = False
CONTEXT_LEN = 2048
HIDDEN_SIZE = 2048
FFN_HIDDEN_SIZE = 3072
NUM_HEADS = 16
VOCAB_SIZE = 50304
NUM_LAYERS = 24
assert HIDDEN_SIZE % NUM_HEADS == 0


class VocabEmbedding(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weight = Parameter(
            torch.empty(
                VOCAB_SIZE,
                HIDDEN_SIZE,
                dtype=torch.float32,
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


class QKVLinear(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weight = Parameter(
            torch.empty(
                3 * HIDDEN_SIZE,
                HIDDEN_SIZE,
                dtype=torch.float32,
            )
        )
        self.bias = Parameter(torch.empty(3 * HIDDEN_SIZE, dtype=torch.float32))

    def forward(self, input_):
        output = torch.matmul(input_, self.weight.t())
        return output + self.bias, None


def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


class CoreAttention(torch.nn.Module):
    def __init__(self, layer_num, precision, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_num = max(1, layer_num)
        self.precision = precision
        self.norm_factor = math.sqrt(HIDDEN_SIZE / NUM_HEADS)
        coeff = self.layer_num
        self.norm_factor *= coeff
        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.precision == 16,
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
        new_context_layer_shape = context_layer.size()[:-2] + (context_layer.size(2) * context_layer.size(3),)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class _SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, input: torch.Tensor, scatter_idx: int, gather_idx: int, seq_len_list, is_first_seq_a2a=True
    ) -> torch.Tensor:
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        seq_world_size = dist.get_world_size()

        if is_first_seq_a2a:
            input_list = [t.contiguous() for t in torch.tensor_split(input, seq_world_size, scatter_idx)]
            shape = torch.tensor(input_list[0].shape)
            output_list = []
            for sl in seq_len_list:
                shape[gather_idx] = sl
                output_list.append(torch.empty([x.item() for x in shape], dtype=input.dtype, device=input.device))
        else:
            split_indices = []
            for sl in seq_len_list:
                if len(split_indices) == 0:
                    split_indices.append(sl)
                else:
                    split_indices.append(split_indices[-1] + sl)
            input_list = [t.contiguous() for t in torch.tensor_split(input, split_indices, scatter_idx)]
            shape = torch.tensor(input_list[0].shape)
            shape[scatter_idx] = seq_len_list[dist.get_rank()]
            output_list = [
                torch.empty([x.item() for x in shape], dtype=input.dtype, device=input.device)
                for _ in range(dist.get_world_size())
            ]

        # TODO Use all_to_all_single instead
        dist.all_to_all(output_list, input_list)

        return torch.cat(output_list, dim=gather_idx).contiguous()

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> torch.Tuple[None, torch.Tensor, None, None]:
        return (None, _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx), None, None)


class DistributedAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        local_attention: torch.nn.Module,
        scatter_idx: int = 2,
        gather_idx: int = 0,
    ) -> None:
        super(DistributedAttention, self).__init__()
        self.local_attn = local_attention
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args: Any) -> torch.Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """

        is_prompt = args[1]
        current_seq_len = args[2]

        num_tokens = query.size(0)
        num_tokens_tensor = torch.tensor(num_tokens, device=query.device)
        num_tokens_list = [torch.empty_like(num_tokens_tensor, device=query.device) for _ in range(dist.get_world_size())]
        dist.all_gather(num_tokens_list, num_tokens_tensor)
        num_tokens_list = [t.item() for t in num_tokens_list]

        if is_prompt:
            seq_len_list = num_tokens_list
        else:
            seq_len_tensor = torch.tensor(current_seq_len, device=query.device)
            seq_len_list = [torch.empty_like(seq_len_tensor, device=query.device) for _ in range(dist.get_world_size())]
            dist.all_gather(seq_len_list, seq_len_tensor)
            seq_len_list = [t.item() for t in seq_len_list]

        # TODO Merge three alltoall calls into one
        # in shape : e.g.,  [s/p:h:]
        # The current shape is [seq, batch, head, head_dim].
        a2a1_time = 0
        time0 = time.time()
        query_layer = _SeqAllToAll.apply(query, self.scatter_idx, self.gather_idx, num_tokens_list)
        key_layer = _SeqAllToAll.apply(key, self.scatter_idx, self.gather_idx, seq_len_list)
        value_layer = _SeqAllToAll.apply(value, self.scatter_idx, self.gather_idx, seq_len_list)
        time1 = time.time()
        a2a1_time = (time1 - time0) * 1000

        # out shape : e.g., [s:h/p:]
        context_layer = self.local_attn(query_layer, key_layer, value_layer, args[0])

        a2a2_time = 0
        time0 = time.time()
        output = _SeqAllToAll.apply(context_layer, self.gather_idx, self.scatter_idx, num_tokens_list, False)
        time1 = time.time()
        a2a2_time = (time1 - time0) * 1000

        # out e.g., [s/p::h]
        return output, a2a1_time + a2a2_time, a2a1_time, a2a2_time


class AttnOutputLinear(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weight = Parameter(
            torch.empty(
                HIDDEN_SIZE,
                HIDDEN_SIZE,
                dtype=torch.float32,
            )
        )
        self.bias = Parameter(torch.empty(HIDDEN_SIZE, dtype=torch.float32))

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
    def __init__(self, layer_num, precision, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_num = layer_num
        self.precision = precision
        self.kv_channels = HIDDEN_SIZE // NUM_HEADS
        self.query_key_value = QKVLinear()
        self.core_attention = CoreAttention(self.layer_num, self.precision)
        self.dist_attention = DistributedAttention(self.core_attention)
        self.dense = AttnOutputLinear()

        # Inference key-value memory
        self.inference_key_memory = None
        self.inference_value_memory = None
        self.inference_current_sequence_len = 0

    def forward(self, hidden_states, attention_mask, set_inference_key_value_memory=False, inference_max_sequence_len=None):
        # hidden_states: [sq, b, h]
        if inference_max_sequence_len is not None and inference_max_sequence_len == 0:
            assert hidden_states.size(0) == 0

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        alloc_time = 0
        alloc_time0 = time.time()
        if set_inference_key_value_memory:
            assert inference_max_sequence_len is not None and inference_max_sequence_len >= 0
            self.inference_key_memory = self._allocate_memory(
                inference_max_sequence_len, hidden_states.size(1), hidden_states.dtype, hidden_states.device
            )
            self.inference_value_memory = self._allocate_memory(
                inference_max_sequence_len, hidden_states.size(1), hidden_states.dtype, hidden_states.device
            )
            self.inference_current_sequence_len = 0
        alloc_time1 = time.time()
        alloc_time = (alloc_time1 - alloc_time0) * 1000

        # Some consistency check.
        if inference_max_sequence_len is not None:
            assert self.inference_current_sequence_len <= self.inference_key_memory.size(0)
            assert inference_max_sequence_len == self.inference_key_memory.size(0)
        # This is added for safety. In case inference_max_sequence_len
        # is not provided, make sure there is no potential memory left
        # from previous inference.
        if inference_max_sequence_len is None:
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
        if inference_max_sequence_len is not None:
            # Adjust the range variables.
            start = self.inference_current_sequence_len
            self.inference_current_sequence_len += key_layer.size(0)
            end = self.inference_current_sequence_len
            # Copy key and values.
            self.inference_key_memory[start:end, ...] = key_layer
            self.inference_value_memory[start:end, ...] = value_layer
            key_layer = self.inference_key_memory[:end, ...]
            value_layer = self.inference_value_memory[:end, ...]

        context_layer, a2a_time, a2a1_time, a2a2_time = self.dist_attention(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            set_inference_key_value_memory,
            self.inference_current_sequence_len,
        )

        output, bias = self.dense(context_layer)

        return output, bias, a2a_time, a2a1_time, a2a2_time, alloc_time

    def _allocate_memory(self, inference_max_sequence_len, batch_size, dtype, device):
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            NUM_HEADS,
            HIDDEN_SIZE // NUM_HEADS,
            dtype=dtype,
            device=device,
        )


class MLPLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, skip_bias_add=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.skip_bias_add = skip_bias_add
        self.weight = Parameter(
            torch.empty(
                output_size,
                input_size,
                dtype=torch.float32,
            )
        )
        self.bias = Parameter(torch.empty(output_size, dtype=torch.float32))

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

        if intermediate.size(0) == 0:
            pass
        else:
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
    def __init__(self, layer_num, precision, drop_path_rate, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_num = layer_num
        self.precision = precision
        self.drop_path_rate = drop_path_rate
        # self.input_layernorm = LayerNorm(
        #     HIDDEN_SIZE,
        #     eps=1e-5,
        #     no_persist_layer_norm=False,
        #     sequence_parallel=False,
        #     apply_layernorm_1p=False,
        # )
        self.input_layernorm = MixedFusedLayerNorm(HIDDEN_SIZE, 1e-5, sequence_parallel_enbaled=False)
        # Self attention.
        self.self_attention = Attention(layer_num, self.precision)

        self.hidden_dropout = 0.1
        self.bias_dropout_fusion = True
        self.drop_path = None

        # self.post_attention_layernorm = LayerNorm(
        #     HIDDEN_SIZE,
        #     eps=1e-5,
        #     no_persist_layer_norm=False,
        #     sequence_parallel=False,
        #     apply_layernorm_1p=False,
        # )
        self.post_attention_layernorm = MixedFusedLayerNorm(HIDDEN_SIZE, 1e-5, sequence_parallel_enbaled=False)

        # MLP
        self.mlp = MLP()

        # Set bias+dropout+add fusion grad_enable execution handler.
        TORCH_MAJOR = int(torch.__version__.split(".")[0])
        TORCH_MINOR = int(torch.__version__.split(".")[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad

        self.all2all_time = 0
        self.all2all1_time = 0
        self.all2all2_time = 0
        self.malloc_time = 0

    def forward(self, hidden_states, attention_mask, set_inference_key_value_memory=False, inference_max_sequence_len=None):
        self.all2all_time = 0
        self.all2all1_time = 0
        self.all2all2_time = 0
        self.malloc_time = 0

        layernorm_output = self.input_layernorm(hidden_states)
        attention_output, attention_bias, a2a_time, a2a1_time, a2a2_time, alloc_time = self.self_attention(
            layernorm_output, attention_mask, set_inference_key_value_memory, inference_max_sequence_len
        )

        self.all2all_time += a2a_time
        self.all2all1_time += a2a1_time
        self.all2all2_time += a2a2_time
        self.malloc_time += alloc_time

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
    def __init__(self, precision, drop_path_rate=0.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.precision = precision
        self.drop_path_rate = drop_path_rate
        self.drop_path_rates = [rate.item() for rate in torch.linspace(0, self.drop_path_rate, NUM_LAYERS)]

        def build_layer(layer_num):
            return TransformerLayer(layer_num, self.precision, drop_path_rate=self.drop_path_rates[layer_num - 1])

        self.layers = []
        for i in range(NUM_LAYERS):
            layer_num = i + 1
            self.layers.append(build_layer(layer_num))
        self.layers = torch.nn.ModuleList(self.layers)

        # self.final_layernorm = LayerNorm(
        #     HIDDEN_SIZE,
        #     eps=1e-5,
        #     no_persist_layer_norm=False,
        #     sequence_parallel=False,
        #     apply_layernorm_1p=False,
        # )
        self.final_layernorm = MixedFusedLayerNorm(HIDDEN_SIZE, 1e-5, sequence_parallel_enbaled=False)

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(self, hidden_states, attention_mask, set_inference_key_value_memory=False, inference_max_sequence_len=None):
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
                return x_, layer

            return custom_forward

        all2all_time = 0
        all2all1_time = 0
        all2all2_time = 0
        malloc_time = 0

        if self.precision == 16:
            for ly in range(NUM_LAYERS):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    hidden_states, layer = custom(ly, ly + 1)(
                        hidden_states, attention_mask, set_inference_key_value_memory, inference_max_sequence_len
                    )
                    all2all_time += layer.all2all_time
                    all2all1_time += layer.all2all1_time
                    all2all2_time += layer.all2all2_time
                    malloc_time += layer.malloc_time
        else:
            for ly in range(NUM_LAYERS):
                hidden_states, layer = custom(ly, ly + 1)(
                    hidden_states, attention_mask, set_inference_key_value_memory, inference_max_sequence_len
                )
                all2all_time += layer.all2all_time
                all2all1_time += layer.all2all1_time
                all2all2_time += layer.all2all2_time
                malloc_time += layer.malloc_time

        print(all2all_time)
        print(all2all1_time)
        print(all2all2_time)
        print(malloc_time)

        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class GPTModel(torch.nn.Module):
    def __init__(self, param_pickle_path, precision=32, device="cuda", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.precision = precision
        self.device = torch.device(device)
        if self.precision == 16:
            assert self.device.type == "cuda"

        torch.cuda.set_device(self.device)

        self.embedding = Embedding()
        self.encoder = Transformer(self.precision)

        self._init_model(param_pickle_path)

    def forward(self, batch):
        (enc_input_ids, enc_attn_mask, enc_position_ids, set_inference_key_value_memory, inference_max_sequence_len) = batch
        enc_input_ids = enc_input_ids.to(self.device)
        enc_attn_mask = enc_attn_mask.to(self.device)
        enc_position_ids = enc_position_ids.to(self.device)

        time0 = time.time()

        encoder_input = self.embedding(enc_input_ids, enc_position_ids)
        encoder_output = self.encoder(
            encoder_input, enc_attn_mask, set_inference_key_value_memory[0].item(), inference_max_sequence_len[0].item()
        )
        word_embeddings_weight = self.embedding.word_embeddings.weight
        logits = torch.matmul(encoder_output, word_embeddings_weight.t())

        time1 = time.time()
        print(f"========== Inference time: {(time1-time0)*1000} ms")
        exit()

        return logits.transpose(0, 1).contiguous()

    def _init_model(self, param_pickle_path):
        with open(param_pickle_path, "rb") as file:
            params = pickle.load(file)
        for (key, _), (_, param) in zip(self.named_parameters(), params):
            self.state_dict()[key].copy_(param)
        self.to(self.device)


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


class DistributedGPTModel(torch.nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.master_ip = config["master_ip"]
        self.master_port = config["master_port"]
        self.num_wrk = config["num_wrk"]

        self.gpt_param_path = config["param_path"]
        self.gpt_tokenizer_kernel_path = config["tokenizer_path"]

        self.precision = config["precision"]
        self.devices = config["devices"]

        self.encoder_seq_length = self.num_wrk * CONTEXT_LEN

    def run(self, inputs, max_gen_len):
        self._wrk_comm_queues: List = []
        processes = []
        mp.set_start_method("spawn")
        for idx in range(self.num_wrk):
            info = {}
            info["rank"] = idx
            info["device"] = self.devices[idx]
            input_q = mp.Queue()
            output_q = mp.Queue()
            self._wrk_comm_queues.append([input_q, output_q])
            process = mp.Process(target=self._run_worker, args=(info, [input_q, output_q]))
            processes.append(process)
            process.start()

        resp_sentences, resp_sentences_seg = self._generate(inputs, max_gen_len)

        for process in processes:
            process.join()

        print(resp_sentences, resp_sentences_seg)

    def _generate(self, inputs, max_gen_len):
        tokenizer = GPTTokenizer(self.gpt_tokenizer_kernel_path)

        context_tokens_tensor, context_length_tensor = self._tokenize_batch(tokenizer, inputs, max_gen_len)
        context_length = context_length_tensor.min().item()
        whole_len = context_tokens_tensor.size(1)
        tokens, attention_mask, position_ids = self._get_batch(context_tokens_tensor)

        eod_id = tokenizer.eos_id
        counter = 0

        batch_size = tokens.size(0)
        is_done = torch.zeros([batch_size], device=tokens.device).byte()
        output_logits = None
        all_generated_indices = None  # used to track all generated indices

        if whole_len > self.encoder_seq_length + 1:
            whole_len = self.encoder_seq_length + 1

        while context_length < whole_len:
            # types2use = None
            if counter == 0:
                # Allocate memory for the entire context.
                set_inference_key_value_memory = True
                tokens2use = tokens[:, :context_length]
                positions2use = position_ids[:, :context_length]
            else:
                # Set this to false so the memory is not reallocated.
                set_inference_key_value_memory = False
                tokens2use = tokens[:, context_length - 1].view(batch_size, -1)
                positions2use = position_ids[:, context_length - 1].view(batch_size, -1)

            attention_mask_repeat = torch.concat([attention_mask for _ in range(batch_size)])

            if counter == 0:
                tokens2use_list = None
                positions2use_list = None
                split_indices = []
                remaining_len = context_length
                for i in range(self.num_wrk):
                    if remaining_len >= CONTEXT_LEN:
                        remaining_len -= CONTEXT_LEN
                        split_indices.append((i + 1) * CONTEXT_LEN)
                    else:
                        split_indices.append(i * CONTEXT_LEN + remaining_len)
                        break
                tokens2use_list = [t.contiguous() for t in torch.tensor_split(tokens2use, split_indices, 1)]
                positions2use_list = [t.contiguous() for t in torch.tensor_split(positions2use, split_indices, 1)]
                if len(tokens2use_list) < self.num_wrk:
                    for _ in range(self.num_wrk - len(tokens2use_list)):
                        tokens2use_list.append(torch.empty([batch_size, 0], dtype=tokens2use.dtype))
                        positions2use_list.append(torch.empty([batch_size, 0], dtype=positions2use.dtype))
                # Adjust attention mask.
                attn_mask = attention_mask_repeat[..., 0:context_length, :context_length]
            else:
                wrk_id = int(context_length / CONTEXT_LEN)
                tokens2use_list = [torch.empty([batch_size, 0], dtype=tokens2use.dtype) for _ in range(self.num_wrk)]
                positions2use_list = [torch.empty([batch_size, 0], dtype=positions2use.dtype) for _ in range(self.num_wrk)]
                tokens2use_list[wrk_id] = tokens2use
                positions2use_list[wrk_id] = positions2use
                # Adjust attention mask.
                attn_mask = attention_mask_repeat[..., context_length - 1 : context_length, :context_length]

            set_key_value_array = torch.tensor([set_inference_key_value_memory] * batch_size)
            max_infer_len_array_list = []
            remaining_len = whole_len
            for _ in range(self.num_wrk):
                if remaining_len >= CONTEXT_LEN:
                    remaining_len -= CONTEXT_LEN
                    max_infer_len_array_list.append(torch.tensor([CONTEXT_LEN] * batch_size))
                elif remaining_len > 0:
                    max_infer_len_array_list.append(torch.tensor([remaining_len] * batch_size))
                    remaining_len = 0
                else:
                    max_infer_len_array_list.append(torch.tensor([0] * batch_size))

            for i in range(self.num_wrk):
                tokens2use = tokens2use_list[i]
                positions2use = positions2use_list[i]
                max_infer_len_array = max_infer_len_array_list[i]
                batch = [tokens2use, attn_mask, positions2use, set_key_value_array, max_infer_len_array]
                wrk_input_q = self._wrk_comm_queues[i][0]
                wrk_input_q.put(batch)

            all_outputs = []
            for i in range(self.num_wrk):
                wrk_output_q = self._wrk_comm_queues[i][1]
                all_outputs.append(wrk_output_q.get())
            output = torch.cat(all_outputs, 1)

            assert output is not None
            output = output.float()
            logits = output[:, -1].view(batch_size, -1).contiguous()

            # make sure it won't sample outside the vocab_size range
            logits[:, tokenizer.vocab_size :] = -float("Inf")

            logits = logits.float()
            logits /= 1.0  # temperature_value
            # handle repetition penality
            logits = self._repetition_penalty(logits, 1.2, all_generated_indices)
            logits = self._top_k_logits(logits, top_k=0, top_p=0.9)
            log_probs = F.softmax(logits, dim=-1)
            prev = torch.multinomial(log_probs, num_samples=1).view(-1)

            started = context_length_tensor <= context_length

            # Clamp the predicted out of vocabulary tokens
            prev = torch.clamp(prev, max=tokenizer.vocab_size - 1)
            new_tokens = self._switch(tokens[:, context_length].view(-1), prev, started)

            # Replace sampled tokens w/ done token if EOD has already been sampled
            new_tokens = self._switch(new_tokens, eod_id, is_done)

            # Insert either new predicted or next prompt token
            tokens[:, context_length] = new_tokens

            if output_logits is None:
                output = F.log_softmax(output[:, :context_length, :], 2)
                indices = torch.unsqueeze(tokens[:, 1 : context_length + 1], 2)
                output_logits = torch.gather(output, 2, indices).squeeze(2)
                all_generated_indices = indices[:, :, 0]
            else:
                output = F.log_softmax(output, 2)
                indices = torch.unsqueeze(new_tokens, 1).unsqueeze(2)
                new_output_logits = torch.gather(output, 2, indices).squeeze(2)

                # TODO(rprenger) we're copying output_logits every time.  Should pre-allocate
                output_logits = torch.cat([output_logits, new_output_logits], 1)
                all_generated_indices = torch.cat([all_generated_indices, indices[:, :, 0]], 1)

            done_token = (prev == eod_id).byte() & started.byte()
            is_done = is_done | done_token

            done = torch.all(is_done)

            context_length += 1
            counter += 1
            if done:
                break

        # tokens and output_logits can be used after the while loop.
        resp_sentences = []
        resp_sentences_seg = []

        for i in range(self.num_wrk):
            self._wrk_comm_queues[i][0].put("EXIT")

        decode_tokens = tokens[:, :context_length]
        decode_tokens = decode_tokens.numpy().tolist()
        for decode_token in decode_tokens:
            sentence = tokenizer.ids_to_text(decode_token)
            resp_sentences.append(sentence)

            words = []
            for token in decode_token:
                # Skip any soft prompt pseudo tokens
                if token not in tokenizer.tokenizer.decoder:
                    continue
                word = tokenizer.tokenizer.decoder[token]
                word = bytearray([tokenizer.tokenizer.byte_decoder[c] for c in word]).decode("utf-8", errors="replace")
                words.append(word)
            resp_sentences_seg.append(words)

        return resp_sentences, resp_sentences_seg

    def _run_worker(self, info, comm_queues: List[mp.Queue]):
        rank = info["rank"]
        device = info["device"]

        model = GPTModel(self.gpt_param_path, self.precision, device)
        model.eval()

        dist.init_process_group(
            "nccl", init_method=f"tcp://{self.master_ip}:{self.master_port}", rank=rank, world_size=self.num_wrk
        )
        dist.barrier()

        while True:
            batch = comm_queues[0].get()
            if isinstance(batch, str) and batch == "EXIT":
                return
            with torch.no_grad():
                output = model(batch)
            comm_queues[1].put(output.cpu())

    def _pad_batch(self, batch, pad_id, max_len):
        context_lengths = []
        max_context_length = max([len(tokens) for tokens in batch])
        for tokens in batch:
            context_length = len(tokens)
            if context_length < max_context_length + max_len:
                tokens.extend([pad_id] * (max_context_length + max_len - context_length))
            context_lengths.append(context_length)
        return batch, context_lengths

    def _tokenize_batch(self, tokenizer, sentences, max_len, add_BOS=True):
        if add_BOS:
            context_tokens = [[tokenizer.eos_id] + tokenizer.text_to_ids(s) for s in sentences]
        else:
            context_tokens = [tokenizer.text_to_ids(s) for s in sentences]
        context_tokens, context_lengths = self._pad_batch(context_tokens, tokenizer.eos_id, max_len)
        context_tokens_tensor = torch.LongTensor(context_tokens)
        context_length_tensor = torch.LongTensor(context_lengths)
        return context_tokens_tensor, context_length_tensor

    def _get_ltor_masks_and_position_ids(self, tokens):
        """Build masks and position id for left to right model."""

        # Extract batch size and sequence length.
        micro_batch_size, seq_length = tokens.size()

        # Attention mask (lower triangular).
        att_mask_batch = 1
        attention_mask = torch.tril(torch.ones((att_mask_batch, seq_length, seq_length), device=tokens.device)).view(
            att_mask_batch, 1, seq_length, seq_length
        )

        # Position ids.
        position_ids = torch.arange(seq_length, dtype=torch.long, device=tokens.device)
        position_ids = position_ids.unsqueeze(0).repeat(micro_batch_size, 1)

        # Convert attention mask to binary:
        attention_mask = attention_mask < 0.5

        return attention_mask, position_ids

    def _get_batch(self, context_tokens):
        """Generate batch from context tokens."""
        # Move to GPU.
        tokens = context_tokens.contiguous()  # .cuda()
        # Get the attention mask and postition ids.
        attention_mask, position_ids = self._get_ltor_masks_and_position_ids(tokens)

        return tokens, attention_mask, position_ids

    def _repetition_penalty(self, logits, repetition_penalty, used_tokens):
        """Implement the repetition penalty, check paper
        https://arxiv.org/pdf/1909.05858.pdf
        """
        if used_tokens is not None and repetition_penalty != 1.0:
            logits_update = torch.gather(logits, 1, used_tokens)
            logits = torch.scatter(logits, 1, used_tokens, logits_update / repetition_penalty)
        return logits

    def _top_k_logits(self, logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
        """This function has been mostly taken from huggingface conversational
        ai code at
            https://medium.com/huggingface/how-to-build-a-state-of-the-art-
                conversational-ai-with-transfer-learning-2d818ac26313"""

        if top_k > 0:
            # Remove all tokens with a probability less than the
            # last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            # Cconvert to 1D
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token
            # above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            for i in range(sorted_indices.size(0)):
                indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                logits[i][indices_to_remove] = filter_value

        return logits

    def _switch(self, val1, val2, boolean):
        boolean = boolean.type_as(val1)
        return (1 - boolean) * val1 + boolean * val2


if __name__ == "__main__":
    config = {}
    config["master_ip"] = "127.0.0.1"
    config["master_port"] = 34565
    config["param_path"] = "/u/qxc4fh/zeyu_workspace/gpt_params.pkl"
    config["tokenizer_path"] = "/u/qxc4fh/zeyu_workspace/gpt_tokenizer_kernel.pkl"
    config["precision"] = 16
    config["devices"] = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    config["num_wrk"] = len(config["devices"])

    test_str = "test " * 2
    test_str = test_str.strip()
    # print(test_str)
    # exit()

    DistributedGPTModel(config).run([test_str], 2048 * 4 + 1)
