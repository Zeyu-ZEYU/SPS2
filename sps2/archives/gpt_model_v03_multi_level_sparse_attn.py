#! /usr/bin/env python3


import argparse
import math
import pickle
import subprocess
import threading
from contextlib import nullcontext
from typing import Any, List, Optional

import einops
import torch
import torch.distributed as dist
import torch.nn.functional as F
from apex.normalization.fused_layer_norm import MixedFusedLayerNorm
from torch.nn.parameter import Parameter
from zutils import net as znet

from megatron import core
from megatron.core.utils import divide

# from megatron.model import LayerNorm
from megatron.model.enums import AttnMaskType
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.fused_softmax import FusedScaleMaskSoftmax

IPS = ["127.0.0.1"]
GPUS = [[0, 1]]
MASTER_DIST_PORT = 43214
MASTER_SERVER_PORT = 43211
NODE_SERVER_PORT = 34119
CODE_NAME_FOR_SHELL = "gpt_model_v03_mlsa_w_pipe.py"
CODE_PATH_FOR_SHELL = "/u/qxc4fh/zeyu_workspace/Megatron-DeepSpeed/sps3"


# Inference config
CONFIG = {}
CONFIG["param_path"] = "/u/qxc4fh/zeyu_workspace/gpt_params.pkl"
CONFIG["tokenizer_path"] = "/u/qxc4fh/zeyu_workspace/gpt_tokenizer_kernel.pkl"
CONFIG["precision"] = 16


# These hyper-parameters are noly for GPTModel rather than DistributedGPTModel
IS_TRAINING = False
CONTEXT_LEN = 2048
HIDDEN_SIZE = 2048
FFN_HIDDEN_SIZE = 3072
NUM_HEADS = 16
VOCAB_SIZE = 50304
NUM_LAYERS = 24
assert HIDDEN_SIZE % NUM_HEADS == 0


# Other global variables
MUL_LEV_SPA_ATN_GRP_SIZES = [1, 2]
MUL_LEV_SPA_ATN_SPARSE_DEGREES = [1, 4]
assert len(MUL_LEV_SPA_ATN_GRP_SIZES) == len(MUL_LEV_SPA_ATN_SPARSE_DEGREES)
_MUL_LEV_SPA_ATN_GRPS = []


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

        tf_shared_dict = args[1]
        is_prompt = tf_shared_dict["set_inference_key_value_memory"]
        num_tokens_list = tf_shared_dict["num_tokens"]

        if is_prompt:
            kv_seq_len_list = num_tokens_list
        else:
            kv_seq_len_list = tf_shared_dict["infer_current_seq_len"]

        # TODO Merge three alltoall calls into one
        # in shape : e.g.,  [s/p:h:]
        # The current shape is [seq, batch, head, head_dim].
        query_layer = _SeqAllToAll.apply(query, self.scatter_idx, self.gather_idx, num_tokens_list)
        key_layer = _SeqAllToAll.apply(key, self.scatter_idx, self.gather_idx, kv_seq_len_list)
        value_layer = _SeqAllToAll.apply(value, self.scatter_idx, self.gather_idx, kv_seq_len_list)

        # out shape : e.g., [s:h/p:]
        context_layer = self.local_attn(query_layer, key_layer, value_layer, args[0])

        output = _SeqAllToAll.apply(context_layer, self.gather_idx, self.scatter_idx, num_tokens_list, False)

        # out e.g., [s/p::h]
        return output


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

    def forward(
        self,
        hidden_states,
        attention_mask,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
        tf_shared_dict=None,
    ):
        # hidden_states: [sq, b, h]
        if inference_max_sequence_len is not None and inference_max_sequence_len == 0:
            assert hidden_states.size(0) == 0

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        if set_inference_key_value_memory:
            assert inference_max_sequence_len is not None and inference_max_sequence_len >= 0
            self.inference_key_memory = self._allocate_memory(
                inference_max_sequence_len, hidden_states.size(1), hidden_states.dtype, hidden_states.device
            )
            self.inference_value_memory = self._allocate_memory(
                inference_max_sequence_len, hidden_states.size(1), hidden_states.dtype, hidden_states.device
            )
            self.inference_current_sequence_len = 0

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

        assert tf_shared_dict is not None
        if "num_tokens" not in tf_shared_dict:
            num_tokens = query_layer.size(0)
            all2all_tensor = torch.tensor([num_tokens, self.inference_current_sequence_len], device=query_layer.device)
            all2all_list = [torch.empty_like(all2all_tensor, device=query_layer.device) for _ in range(dist.get_world_size())]
            dist.all_gather(all2all_list, all2all_tensor)
            num_tokens_list = []
            infer_current_seq_len_list = []
            tf_shared_dict["num_tokens"] = num_tokens_list
            tf_shared_dict["infer_current_seq_len"] = infer_current_seq_len_list
            for a2a_tensor in all2all_list:
                num_tokens_list.append(a2a_tensor[0].item())
                infer_current_seq_len_list.append(a2a_tensor[1].item())
            tf_shared_dict["set_inference_key_value_memory"] = set_inference_key_value_memory

        context_layer = self.dist_attention(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            tf_shared_dict,
        )

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


class MLSASoftmax(torch.nn.Module):
    def __init__(self, input_in_fp16, mask_func, scale, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_in_fp16 = input_in_fp16
        self.mask_func = mask_func
        self.scale = scale

    def forward(self, input, mask):
        if self.input_in_fp16:
            input = input.float()
        input = input * self.scale
        # shape (batch, head, q, k)
        mask_output = self.mask_func(input, mask)

        exp_mo = torch.exp(mask_output)
        denomi = torch.sum(exp_mo, dim=-1, keepdim=True)
        probs = exp_mo / denomi

        # probs = torch.nn.Softmax(dim=-1)(mask_output)

        if self.input_in_fp16:
            probs = probs.half()

        # denomi must be float32
        return probs, denomi


class MLSACoreAttention(torch.nn.Module):
    def __init__(self, layer_num, precision, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_num = max(1, layer_num)
        self.precision = precision
        self.norm_factor = math.sqrt(HIDDEN_SIZE / NUM_HEADS)
        coeff = self.layer_num
        self.norm_factor *= coeff
        self.scale_mask_softmax = MLSASoftmax(precision == 16, attention_mask_func, coeff)
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
        attention_probs, denomi = self.scale_mask_softmax(attention_scores, attention_mask)

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

        return context_layer, denomi


class _MLSASeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, input: torch.Tensor, scatter_idx: int, gather_idx: int, seq_len_list, group, is_first_seq_a2a=True
    ) -> torch.Tensor:
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        seq_world_size = dist.get_world_size(group)

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
            shape[scatter_idx] = seq_len_list[dist.get_rank(group)]
            output_list = [
                torch.empty([x.item() for x in shape], dtype=input.dtype, device=input.device)
                for _ in range(dist.get_world_size(group))
            ]

        # TODO Use all_to_all_single instead
        dist.all_to_all(output_list, input_list, group=group)

        return torch.cat(output_list, dim=gather_idx).contiguous()

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> torch.Tuple[None, torch.Tensor, None, None]:
        return (None, _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx), None, None)


class MLSAForOneLevel(torch.nn.Module):
    def __init__(self, level, core_attn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.level = level
        self.core_attn = core_attn
        self.group_size = MUL_LEV_SPA_ATN_GRP_SIZES[level]
        self.sparse_degree = MUL_LEV_SPA_ATN_SPARSE_DEGREES[level]
        self.group = None

    def forward(self, q, k, v, mask, tf_shared_dict, is_prompt=True):
        if self.group is None:
            self.group = _MUL_LEV_SPA_ATN_GRPS[self.level]

        # get all indices of sampled tokens
        assert tf_shared_dict is not None
        is_first_layer = False
        if f"{self.level}:indices" not in tf_shared_dict:
            is_first_layer = True
            # None placeholder when sparse degree is 1.
            tf_shared_dict[f"{self.level}:indices"] = None
            # num_tokens = q.size(0)
            if self.sparse_degree > 1:
                batch_size = q.size(1)
                num_heads = q.size(2)
                head_size = q.size(3)
                indices_for_heads = []
                for hd in range(num_heads):
                    start = hd % self.sparse_degree
                    indices = [
                        x for x in range(start, k.size(0) // self.sparse_degree * self.sparse_degree, self.sparse_degree)
                    ]
                    indices_for_heads.append(indices)
                indices_tensor = torch.tensor(indices_for_heads, dtype=torch.long, device=q.device)
                indices_tensor = einops.rearrange(indices_tensor, "h t -> t h")
                indices_tensor = einops.repeat(indices_tensor, "t h -> t b h r", b=batch_size, r=head_size)
                tf_shared_dict[f"{self.level}:indices"] = indices_for_heads
                tf_shared_dict[f"{self.level}:indices_ts"] = indices_tensor
                # Create recover-indices
                # The complete indices will be computed during context_layer recovering
                rc_indices_tensor = torch.empty([k.size(0), NUM_HEADS], dtype=torch.long, device=q.device).fill_(
                    indices_tensor.size(0)
                )
                for i in range(indices_tensor.size(0)):
                    for h in range(NUM_HEADS):
                        token_idx = i * self.sparse_degree + h % self.sparse_degree
                        rc_indices_tensor[token_idx][h] = i
                tf_shared_dict[f"{self.level}:rc_indices_ts"] = rc_indices_tensor
        else:
            if self.sparse_degree > 1:
                indices_tensor = tf_shared_dict[f"{self.level}:indices_ts"]
                rc_indices_tensor = tf_shared_dict[f"{self.level}:rc_indices_ts"]

        if self.sparse_degree > 1:
            k_ = k.gather(0, indices_tensor)
            v_ = v.gather(0, indices_tensor)
            if is_prompt:
                q_ = q.gather(0, indices_tensor)
            else:
                q_ = q
        else:
            q_ = q
            k_ = k
            v_ = v

        num_input_tokens = q_.size(0)
        num_kv_tokens = k_.size(0)

        if is_first_layer:
            all2all_tensor = torch.tensor([num_input_tokens, num_kv_tokens], device=q_.device)
            all2all_list = [torch.empty_like(all2all_tensor, device=q_.device) for _ in range(dist.get_world_size(self.group))]
            dist.all_gather(all2all_list, all2all_tensor, group=self.group)
            num_input_tokens_list = []
            num_kv_tokens_list = []
            tf_shared_dict[f"{self.level}:num_in_tk"] = num_input_tokens_list
            tf_shared_dict[f"{self.level}:num_kv_tk"] = num_kv_tokens_list
            for a2a_tensor in all2all_list:
                num_input_tokens_list.append(a2a_tensor[0].item())
                num_kv_tokens_list.append(a2a_tensor[1].item())
        else:
            num_input_tokens_list = tf_shared_dict[f"{self.level}:num_in_tk"]
            num_kv_tokens_list = tf_shared_dict[f"{self.level}:num_kv_tk"]

        # TODO Merge three alltoall calls into one
        # in shape : e.g.,  [s/p:h:]
        # The current shape is [seq, batch, head, head_dim].
        query_layer = _MLSASeqAllToAll.apply(q_, 2, 0, num_input_tokens_list, self.group)
        key_layer = _MLSASeqAllToAll.apply(k_, 2, 0, num_kv_tokens_list, self.group)
        value_layer = _MLSASeqAllToAll.apply(v_, 2, 0, num_kv_tokens_list, self.group)

        # create attn mask for MLSA
        if is_first_layer:
            bs = query_layer.size(1)
            nint = query_layer.size(0)
            nkvt = key_layer.size(0)
            attn_mask = torch.tril(torch.ones((1, nkvt, nkvt), device=query_layer.device)).view(1, 1, nkvt, nkvt)
            attn_mask = attn_mask < 0.5
            attn_mask = torch.concat([attn_mask for _ in range(bs)])
            attn_mask = attn_mask[..., nkvt - nint : nkvt, :nkvt]
            tf_shared_dict[f"{self.level}:mask"] = attn_mask
        else:
            attn_mask = tf_shared_dict[f"{self.level}:mask"]

        if query_layer.size(0) == 0 or key_layer.size(0) == 0:
            context_layer = torch.zeros(
                query_layer.size(0),
                query_layer.size(1),
                query_layer.size(2) * query_layer.size(3),
                dtype=query_layer.dtype,
                device=query_layer.device,
            )
            denomi = torch.zeros(
                [query_layer.size(1), query_layer.size(2), query_layer.size(0), 1],
                dtype=torch.float32,
                device=query_layer.device,
            )
        else:
            # out shape : e.g., [s:h/p:]
            context_layer, denomi = self.core_attn(query_layer, key_layer, value_layer, attn_mask)

        output = _MLSASeqAllToAll.apply(context_layer, 0, 2, num_input_tokens_list, self.group, False)

        # all2all denominator
        denomi = einops.rearrange(denomi, "b h t x -> t b h x")
        split_indices = []
        for sl in num_input_tokens_list:
            if len(split_indices) == 0:
                split_indices.append(sl)
            else:
                split_indices.append(split_indices[-1] + sl)
        input_list = [t.contiguous() for t in torch.tensor_split(denomi, split_indices, 0)]
        shape = torch.tensor(input_list[0].shape)
        shape[0] = num_input_tokens_list[dist.get_rank(self.group)]
        output_list = [
            torch.empty([x.item() for x in shape], dtype=denomi.dtype, device=denomi.device)
            for _ in range(dist.get_world_size(self.group))
        ]
        dist.all_to_all(output_list, input_list, group=self.group)
        denomi = torch.cat(output_list, dim=2).contiguous()
        # denomi = einops.rearrange(denomi, "t b h x -> b h t x")

        output = output.view(output.size(0), output.size(1), q_.size(2), q_.size(3))

        if self.sparse_degree > 1:
            if is_prompt:
                # recover output tensor with inserting zeros
                output_shape = torch.tensor(output.shape)
                output_shape[0] = 1
                added_tensor = torch.zeros(*output_shape, dtype=output.dtype, device=output.device)
                output = torch.cat([output, added_tensor])
                rc_indices_tensor = einops.repeat(rc_indices_tensor, "t h -> t b h r", b=output.size(1), r=output.size(3))
                output = output.gather(0, rc_indices_tensor)

                # convert denominator for later use
                output_shape[3] = 1
                added_tensor = torch.zeros(*output_shape, dtype=output.dtype, device=output.device)
                denomi = torch.cat([denomi, added_tensor])
                rc_indices_tensor = tf_shared_dict[f"{self.level}:rc_indices_ts"]
                rc_indices_tensor = einops.repeat(rc_indices_tensor, "t h -> t b h r", b=output.size(1), r=1)
                denomi = denomi.gather(0, rc_indices_tensor)

        return output, denomi


class MLSAttention(torch.nn.Module):
    def __init__(self, layer_num, precision, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_num = layer_num
        self.precision = precision
        self.kv_channels = HIDDEN_SIZE // NUM_HEADS
        self.query_key_value = QKVLinear()
        # self.core_attention = CoreAttention(self.layer_num, self.precision)
        # self.dist_attention = DistributedAttention(self.core_attention)
        self.core_attention = MLSACoreAttention(self.layer_num, self.precision)
        self.mlsa_attns = []
        for lev in range(len(MUL_LEV_SPA_ATN_GRP_SIZES)):
            self.mlsa_attns.append(MLSAForOneLevel(lev, self.core_attention))
        self.mlsa_attns = torch.nn.ModuleList(self.mlsa_attns)
        self.dense = AttnOutputLinear()

        # Inference key-value memory
        self.inference_key_memory = None
        self.inference_value_memory = None
        self.inference_current_sequence_len = 0

    def forward(
        self,
        hidden_states,
        attention_mask,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
        tf_shared_dict=None,
    ):
        # hidden_states: [sq, b, h]
        if inference_max_sequence_len is not None and inference_max_sequence_len == 0:
            assert hidden_states.size(0) == 0

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        if set_inference_key_value_memory:
            assert inference_max_sequence_len is not None and inference_max_sequence_len >= 0
            self.inference_key_memory = self._allocate_memory(
                inference_max_sequence_len, hidden_states.size(1), hidden_states.dtype, hidden_states.device
            )
            self.inference_value_memory = self._allocate_memory(
                inference_max_sequence_len, hidden_states.size(1), hidden_states.dtype, hidden_states.device
            )
            self.inference_current_sequence_len = 0

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

        # context_layer = self.dist_attention(
        #     query_layer,
        #     key_layer,
        #     value_layer,
        #     attention_mask,
        #     tf_shared_dict,
        # )
        context_layers = []
        denomis = []
        sum_denomis = None
        for mlsa in self.mlsa_attns:
            context_layer, denomi = mlsa(
                query_layer, key_layer, value_layer, attention_mask, tf_shared_dict, set_inference_key_value_memory
            )
            context_layers.append(context_layer)
            denomis.append(denomi)
            if sum_denomis is None:
                sum_denomis = denomi
            else:
                sum_denomis += denomi
            torch.cuda.synchronize()
        sum_cxt_layers = None
        for ctx_ly, deno in zip(context_layers, denomis):
            alpha = torch.div(deno, sum_denomis)
            alpha = torch.nan_to_num(alpha)
            scaled_ctx_ly = ctx_ly * alpha
            if sum_cxt_layers is None:
                sum_cxt_layers = scaled_ctx_ly
            else:
                sum_cxt_layers += scaled_ctx_ly
        context_layer = sum_cxt_layers.view(
            sum_cxt_layers.size(0), sum_cxt_layers.size(1), sum_cxt_layers.size(2) * sum_cxt_layers.size(3)
        )

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
        self.self_attention = MLSAttention(layer_num, self.precision)

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

    def forward(
        self,
        hidden_states,
        attention_mask,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
        tf_shared_dict=None,
    ):
        layernorm_output = self.input_layernorm(hidden_states)
        attention_output, attention_bias = self.self_attention(
            layernorm_output, attention_mask, set_inference_key_value_memory, inference_max_sequence_len, tf_shared_dict
        )
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

        self.tf_shared_dict = {}

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
                return x_

            return custom_forward

        self.tf_shared_dict = {}
        if self.precision == 16:
            for ly in range(NUM_LAYERS):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    hidden_states = custom(ly, ly + 1)(
                        hidden_states,
                        attention_mask,
                        set_inference_key_value_memory,
                        inference_max_sequence_len,
                        self.tf_shared_dict,
                    )
        else:
            for ly in range(NUM_LAYERS):
                hidden_states = custom(ly, ly + 1)(
                    hidden_states,
                    attention_mask,
                    set_inference_key_value_memory,
                    inference_max_sequence_len,
                    self.tf_shared_dict,
                )

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
        encoder_input = self.embedding(enc_input_ids, enc_position_ids)
        encoder_output = self.encoder(
            encoder_input, enc_attn_mask, set_inference_key_value_memory[0].item(), inference_max_sequence_len[0].item()
        )
        word_embeddings_weight = self.embedding.word_embeddings.weight
        logits = torch.matmul(encoder_output, word_embeddings_weight.t())
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
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        num_wrks = 0
        for node in GPUS:
            for _ in node:
                num_wrks += 1
        self.num_wrks = num_wrks
        self.encoder_seq_length = self.num_wrks * CONTEXT_LEN

        self.model_remote_conns = {}
        self.model_local_conn = None

    def run(self, inputs, max_gen_len):
        master_server_listener = znet.SocketMsger.tcp_listener("0.0.0.0", MASTER_SERVER_PORT)

        def run_listener_thread(listener, model_conns, num_wrks):
            while True:
                conn, _ = listener.accept()
                rank = conn.recv()
                model_conns[rank] = conn
                if len(model_conns) == num_wrks:
                    break

        list_thread = threading.Thread(
            target=run_listener_thread, args=(master_server_listener, self.model_remote_conns, self.num_wrks)
        )
        list_thread.start()
        threading.Thread(
            target=self._run_core,
            args=(
                0,
                True,
            ),
        ).start()
        list_thread.join()
        for i in range(self.num_wrks):
            self.model_remote_conns[i].send("START")

        resp_sentences, resp_sentences_seg = self._generate(inputs, max_gen_len)

        print(resp_sentences, resp_sentences_seg)

    def _run_core(self, rank, is_master=False):
        world_size = self.num_wrks

        if is_master:
            assert rank == 0
            listener_ip = "127.0.0.1"
        else:
            listener_ip = IPS[0]

        gpus_idx = 0
        gpu_id = None
        for node in GPUS:
            for gid in node:
                if gpus_idx == rank:
                    gpu_id = gid
                    break
                gpus_idx += 1
            if gpu_id is not None:
                break
        assert gpu_id is not None

        self.model_local_conn = znet.SocketMsger.tcp_connect(listener_ip, MASTER_SERVER_PORT)
        self.model_local_conn.send(rank)
        model = GPTModel(CONFIG["param_path"], CONFIG["precision"], f"cuda:{gpu_id}")
        model.eval()
        # get START cmd
        self.model_local_conn.recv()
        dist.init_process_group("nccl", init_method=f"tcp://{IPS[0]}:{MASTER_DIST_PORT}", rank=rank, world_size=world_size)

        for gsize in MUL_LEV_SPA_ATN_GRP_SIZES:
            start = 0
            end = gsize
            for i in range(0, world_size, gsize):
                start = i
                end = i + gsize
                if end > world_size:
                    end = world_size
                group_ids = [r for r in range(start, end)]
                group = dist.new_group(group_ids)
                if rank in group_ids:
                    _MUL_LEV_SPA_ATN_GRPS.append(group)

        while True:
            batch = self.model_local_conn.recv()
            if isinstance(batch, str) and batch == "EXIT":
                self.model_local_conn.close()
                return
            with torch.no_grad():
                output = model(batch)
            self.model_local_conn.send(output.cpu())

    def _generate(self, inputs, max_gen_len):
        tokenizer = GPTTokenizer(CONFIG["tokenizer_path"])

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
                for i in range(self.num_wrks):
                    if remaining_len >= CONTEXT_LEN:
                        remaining_len -= CONTEXT_LEN
                        split_indices.append((i + 1) * CONTEXT_LEN)
                    else:
                        split_indices.append(i * CONTEXT_LEN + remaining_len)
                        break
                tokens2use_list = [t.contiguous() for t in torch.tensor_split(tokens2use, split_indices, 1)]
                positions2use_list = [t.contiguous() for t in torch.tensor_split(positions2use, split_indices, 1)]
                if len(tokens2use_list) < self.num_wrks:
                    for _ in range(self.num_wrks - len(tokens2use_list)):
                        tokens2use_list.append(torch.empty([batch_size, 0], dtype=tokens2use.dtype))
                        positions2use_list.append(torch.empty([batch_size, 0], dtype=positions2use.dtype))
                # Adjust attention mask.
                attn_mask = attention_mask_repeat[..., 0:context_length, :context_length]
            else:
                wrk_id = int(context_length / CONTEXT_LEN)
                tokens2use_list = [torch.empty([batch_size, 0], dtype=tokens2use.dtype) for _ in range(self.num_wrks)]
                positions2use_list = [torch.empty([batch_size, 0], dtype=positions2use.dtype) for _ in range(self.num_wrks)]
                tokens2use_list[wrk_id] = tokens2use
                positions2use_list[wrk_id] = positions2use
                # Adjust attention mask.
                attn_mask = attention_mask_repeat[..., context_length - 1 : context_length, :context_length]

            set_key_value_array = torch.tensor([set_inference_key_value_memory] * batch_size)
            max_infer_len_array_list = []
            remaining_len = whole_len - 1
            for _ in range(self.num_wrks):
                if remaining_len >= CONTEXT_LEN:
                    remaining_len -= CONTEXT_LEN
                    max_infer_len_array_list.append(torch.tensor([CONTEXT_LEN] * batch_size))
                elif remaining_len > 0:
                    max_infer_len_array_list.append(torch.tensor([remaining_len] * batch_size))
                    remaining_len = 0
                else:
                    max_infer_len_array_list.append(torch.tensor([0] * batch_size))

            for i in range(self.num_wrks):
                tokens2use = tokens2use_list[i]
                positions2use = positions2use_list[i]
                max_infer_len_array = max_infer_len_array_list[i]
                batch = [tokens2use, attn_mask, positions2use, set_key_value_array, max_infer_len_array]
                model_conn = self.model_remote_conns[i]
                model_conn.send(batch)

            all_outputs = []
            for i in range(self.num_wrks):
                model_conn = self.model_remote_conns[i]
                all_outputs.append(model_conn.recv())
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

        for i in range(self.num_wrks):
            self.model_remote_conns[i].send("EXIT")
            self.model_remote_conns[i].close()

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
        # WARNING: REMOVE IT WHEN SUPPORTING LONG SEQUENCES!
        pos_ids = [i % CONTEXT_LEN for i in range(seq_length)]
        position_ids = torch.tensor(pos_ids, dtype=torch.long, device=tokens.device)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--server", action="store_true")
    parser.add_argument("-w", "--worker", action="store_true")
    parser.add_argument("-r", "--rank", default=0, type=int)
    args = parser.parse_args()

    if args.server:
        listener = znet.SocketMsger.tcp_listener("0.0.0.0", NODE_SERVER_PORT)

        while True:
            req_conn, _ = listener.accept()
            cmds = req_conn.recv()
            for cmd in cmds:
                subprocess.call(cmd, shell=True)
            req_conn.close()
    else:
        if args.worker:
            assert args.rank > 0
            dist_gpt_model = DistributedGPTModel()
            dist_gpt_model._run_core(args.rank)
        else:
            remote_rank = 0
            for node_idx in range(len(IPS)):
                ip = IPS[node_idx]
                node_gpus = GPUS[node_idx]
                conn = znet.SocketMsger.tcp_connect(ip, NODE_SERVER_PORT)
                cmds = []
                for _ in node_gpus:
                    if remote_rank == 0:
                        remote_rank += 1
                        continue
                    cmds.append(f"python3 {CODE_PATH_FOR_SHELL}/{CODE_NAME_FOR_SHELL} -w -r {remote_rank}")
                    remote_rank += 1
                conn.send(cmds)
                conn.close()
            dist_gpt_model = DistributedGPTModel()
            dist_gpt_model.run(["How big is the universe?"], 50)
