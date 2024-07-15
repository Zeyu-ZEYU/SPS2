#! /usr/bin/env python3


import argparse
import logging
import os
import time

import psutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.profiler import ProfilerActivity, profile, record_function

# from einops import rearrange
# from flash_attn.flash_attn_interface import flash_attn_varlen_func


RANK_0_IP = "192.168.0.3"
RANK_0_PORT = 31187


BATCH_SIZE = 1
PART_SEQ_LEN = 2048


MODEL_DIM = 7680
NUM_HEADS = 60
assert MODEL_DIM % NUM_HEADS == 0
HEAD_DIM = MODEL_DIM // NUM_HEADS
NUM_LAYERS = 20


# send_stream = torch.cuda.Stream()
# recv_stream = torch.cuda.Stream()
compute_stream = torch.cuda.Stream()


# recv_complete_event = None
compute_complete_event = None


send_handles = []


class Logger(object):
    def __init__(self, job_name, file_path, log_level=logging.INFO, mode="w"):
        self.__logger = logging.getLogger(job_name)
        self.__logger.setLevel(log_level)
        self.__fh = logging.FileHandler(filename=file_path, mode=mode)
        self.__formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
        self.__fh.setFormatter(self.__formatter)
        self.__logger.addHandler(self.__fh)

    @property
    def logger(self):
        return self.__logger


def _singleton_timer(cls):
    timers = {}

    def get_timer(name):
        if name not in timers:
            timers[name] = cls(name)
        return timers[name]

    return get_timer


@_singleton_timer
class _Timer:
    def __init__(self, name) -> None:
        self.name = name
        self.total_time = 0

    def __enter__(self):
        # torch.cuda.synchronize()
        # self.start = time.time()
        return self

    def __exit__(self, *args):
        torch.cuda.synchronize()
        global logger
        logger.info(self.name)
        # self.interval = 1000 * (time.time() - self.start)
        # self.total_time += self.interval
        # print(f"{self.name}: {self.interval}ms out of {self.total_time}ms.")
        pass


def _pause(ms):
    torch.cuda.synchronize()
    time.sleep(ms / 1000)
    torch.cuda.synchronize()


def _all_gather(outputs, input):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    handles = []
    global send_handles
    send_handles = []

    for i in range(world_size):
        if i != rank:
            if i > rank:
                # with torch.cuda.stream(send_stream):
                send_handles.append(dist.isend(input, dst=i, group=first_comm_group))
            else:
                # with torch.cuda.stream(recv_stream):
                handles.append(dist.irecv(outputs[i], src=i, group=first_comm_group))
        else:
            outputs[i] = input

    for i in range(world_size):
        if i != rank:
            if i > rank:
                # with torch.cuda.stream(recv_stream):
                handles.append(dist.irecv(outputs[i], src=i, group=second_comm_group))
            else:
                # with torch.cuda.stream(send_stream):
                send_handles.append(dist.isend(input, dst=i, group=second_comm_group))

    # global recv_complete_event
    # recv_complete_event = torch.cuda.Event()
    # recv_complete_event.record(stream=recv_stream)
    # recv_complete_event.wait()

    for handle in handles:
        handle.wait()

    return outputs


def _reduce_scatter(output, inputs):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    output = torch.zeros_like(output)
    recv_tensors = [torch.empty_like(inputs[rank]) for _ in range(world_size)]
    handles = []

    for i in range(world_size):
        if i != rank:
            if i > rank:
                # with torch.cuda.stream(send_stream):
                send_handles.append(dist.isend(inputs[i], dst=i, group=first_comm_group))
            else:
                # with torch.cuda.stream(recv_stream):
                handles.append(dist.irecv(recv_tensors[i], src=i, group=first_comm_group))
        else:
            recv_tensors[i] = inputs[i]

    for i in range(world_size):
        if i != rank:
            if i > rank:
                # with torch.cuda.stream(recv_stream):
                handles.append(dist.irecv(recv_tensors[i], src=i, group=second_comm_group))
            else:
                # with torch.cuda.stream(send_stream):
                send_handles.append(dist.isend(inputs[i], dst=i, group=second_comm_group))

    for handle in handles:
        handle.wait()

    # global recv_complete_event
    # recv_complete_event = torch.cuda.Event()
    # recv_complete_event.record(stream=recv_stream)
    # recv_complete_event.wait()

    for tsr in recv_tensors:
        output += tsr

    return output


def _init_data():
    data = {}
    data["seq_embed"] = torch.rand([BATCH_SIZE, PART_SEQ_LEN, MODEL_DIM], dtype=torch.float16, device="cuda:0")
    data["p_qkv_gen"] = torch.rand([MODEL_DIM, 3 * MODEL_DIM // args.world_size], dtype=torch.float16, device="cuda:0")
    data["p_attn_linear"] = torch.rand([MODEL_DIM // args.world_size, MODEL_DIM], dtype=torch.float16, device="cuda:0")
    data["p_mlp_1"] = torch.rand([MODEL_DIM, 4 * MODEL_DIM // args.world_size], dtype=torch.float16, device="cuda:0")
    data["p_mlp_2"] = torch.rand([4 * MODEL_DIM // args.world_size, MODEL_DIM], dtype=torch.float16, device="cuda:0")
    data["f_attn"] = (
        torch.nn.MultiheadAttention(MODEL_DIM // args.world_size, NUM_HEADS_PER_PART, 0.1, batch_first=True)
        .to(torch.float16)
        .to("cuda:0")
    )
    data["f_layernorm"] = torch.nn.LayerNorm(MODEL_DIM).to(torch.float16).to("cuda:0")
    data["f_dropout"] = torch.nn.Dropout(0.1)
    data["f_gelu"] = torch.nn.GELU()

    return data


def _run_layer():
    seq_embed = data["seq_embed"]
    p_qkv_gen = data["p_qkv_gen"]
    p_attn_linear = data["p_attn_linear"]
    p_mlp_1 = data["p_mlp_1"]
    p_mlp_2 = data["p_mlp_2"]
    f_attn = data["f_attn"]
    f_layernorm = data["f_layernorm"]
    f_dropout = data["f_dropout"]
    f_gelu = data["f_gelu"]

    with _Timer("layernorm_dropout"):
        seq_embed = f_layernorm(seq_embed)

    with _Timer("attn_all_gather"):
        seq_embed_list = [torch.empty_like(seq_embed, dtype=torch.float16, device="cuda:0") for _ in range(args.world_size)]
        dist.all_gather(seq_embed_list, seq_embed)
        seq_embed = torch.concat(seq_embed_list, dim=1)

    with _Timer("attn"):
        qkv = torch.matmul(seq_embed, p_qkv_gen)
        PART_DIM = NUM_HEADS_PER_PART * HEAD_DIM
        q = qkv[..., :PART_DIM]
        k = qkv[..., PART_DIM : 2 * PART_DIM]
        v = qkv[..., 2 * PART_DIM :]

        # q_len = q.size(1)
        # k_len = k.size(1)
        # q = rearrange(q, "b s (h d) -> (b s) h d", h=NUM_HEADS_PER_PART)
        # k = rearrange(k, "b s (h d) -> (b s) h d", h=NUM_HEADS_PER_PART)
        # v = rearrange(v, "b s (h d) -> (b s) h d", h=NUM_HEADS_PER_PART)

        # # attn
        # cu_seqlens_q = torch.arange(0, (BATCH_SIZE + 1) * q_len, step=q_len, dtype=torch.int32, device="cuda:0")
        # cu_seqlens_k = torch.arange(0, (BATCH_SIZE + 1) * k_len, step=k_len, dtype=torch.int32, device="cuda:0")
        # attn_output = flash_attn_varlen_func(
        #     q,
        #     k,
        #     v,
        #     cu_seqlens_q,
        #     cu_seqlens_k,
        #     q_len,
        #     k_len,
        #     0.1,
        #     causal=True,
        # )
        attn_output, _ = f_attn(q, k, v)

        # attn_output = rearrange(attn_output, "(b s) h d -> b s (h d)", b=BATCH_SIZE)
        output = torch.matmul(attn_output, p_attn_linear)

    with _Timer("attn_reduce_scatter"):
        rs_inputs = list(torch.tensor_split(output, args.world_size, dim=1))
        rs_output = torch.empty_like(rs_inputs[0], dtype=torch.float16, device="cuda:0")
        dist.reduce_scatter(rs_output, rs_inputs)

    with _Timer("layernorm_dropout"):
        seq_embed = f_dropout(rs_output)

        seq_embed = f_layernorm(seq_embed)

    # MLP
    with _Timer("mlp_all_gather"):
        seq_embed_list = [torch.empty_like(seq_embed, dtype=torch.float16, device="cuda:0") for _ in range(args.world_size)]
        dist.all_gather(seq_embed_list, seq_embed)
        seq_embed = torch.concat(seq_embed_list, dim=1)

    with _Timer("mlp"):
        mlp_1_output = torch.matmul(seq_embed, p_mlp_1)
        gelu_result = f_gelu(mlp_1_output)
        mlp_2_output = torch.matmul(gelu_result, p_mlp_2)

    with _Timer("mlp_reduce_scatter"):
        rs_inputs = list(torch.tensor_split(mlp_2_output, args.world_size, dim=1))
        rs_output = torch.empty_like(rs_inputs[0], dtype=torch.float16, device="cuda:0")
        dist.reduce_scatter(rs_output, rs_inputs)

    with _Timer("layernorm_dropout"):
        seq_embed = f_dropout(rs_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rank", default=0, type=int)
    parser.add_argument("-g", "--gpu_id", default=0, type=int)
    parser.add_argument("-w", "--world_size", default=1, type=int)
    args = parser.parse_args()

    assert NUM_HEADS % args.world_size == 0
    NUM_HEADS_PER_PART = NUM_HEADS // args.world_size

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"
    dist.init_process_group("nccl", init_method=f"tcp://{RANK_0_IP}:{RANK_0_PORT}", rank=args.rank, world_size=args.world_size)

    first_comm_group = dist.new_group(backend="nccl")
    second_comm_group = dist.new_group(backend="nccl")

    data = _init_data()
    logger = Logger(job_name=f"rank{args.rank}", file_path=f"/home/zeyu/tests/logs/rank{args.rank}.log").logger

    def run_bw_recoder(q):
        bw_logger = Logger("BW", f"/home/zeyu/tests/logs/netif_rank{args.rank}.log").logger
        recv0 = psutil.net_io_counters(pernic=True)["enp161s0f0np0"].bytes_recv
        sent0 = psutil.net_io_counters(pernic=True)["enp161s0f0np0"].bytes_sent
        time0 = time.time()
        while True:
            time.sleep(0.003)
            recv1 = psutil.net_io_counters(pernic=True)["enp161s0f0np0"].bytes_recv
            sent1 = psutil.net_io_counters(pernic=True)["enp161s0f0np0"].bytes_sent
            time1 = time.time()
            time_diff = time1 - time0
            bw_in = (recv1 - recv0) / time_diff / 1048576
            bw_out = (sent1 - sent0) / time_diff / 1048576
            bw_logger.info(f"{bw_in} {bw_out}")
            recv0 = recv1
            sent0 = sent1
            time0 = time1
            try:
                q.get(False)
                break
            except Exception:
                continue

    mp_queue = mp.Queue()
    rcd_p = mp.Process(target=run_bw_recoder, args=(mp_queue,))
    rcd_p.start()

    time.sleep(3)

    dist.barrier()
    with _Timer("end_to_end_time"):

        # def trace_handler(p):
        #     p.export_chrome_trace("trace.json")

        # with profile(
        #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #     schedule=torch.profiler.schedule(wait=0, warmup=1, active=NUM_LAYERS - 1),
        #     on_trace_ready=trace_handler,
        # ) as p:
        for ly in range(NUM_LAYERS):
            # if ly == NUM_LAYERS - 5:
            #     for handle in send_handles:
            #         handle.wait()
            #     dist.barrier()
            _run_layer()
            # p.step()

    time.sleep(1)
    mp_queue.put("END")
    rcd_p.join()
    # print("press ctrl-c to terminate it.")
