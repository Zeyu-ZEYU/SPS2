#!/usr/bin/env python3


import time

import torch


class Timer:
    def __init__(self, label="Timer") -> None:
        self.label = label

    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.time()
        return self

    def __exit__(self, *args):
        torch.cuda.synchronize()
        self.interval = 1000 * (time.time() - self.start)
        print(f"{self.label}: {self.interval}ms.")


q = torch.ones([40, 1, 128], dtype=torch.float16, device="cuda:0")
quant_k = torch.ones([40, 128, 64000], dtype=torch.uint8, device="cuda:0")
quant_v = torch.ones([40, 64000, 128], dtype=torch.uint8, device="cuda:0")
min = 1.4
max = 8.3
scaling = (max - min) / 16
sm = torch.nn.Softmax(dim=2)


for _ in range(10):
    with Timer("decompress"):
        k = (quant_k * scaling + min).to(torch.float16)
    with Timer("bmm"):
        torch.bmm(q, k)

    with Timer("convert_to_fp16"):
        quant_k_ = quant_k.to(torch.float16)
    with Timer("bmm compress"):
        r = torch.bmm(q, quant_k_)
    with Timer("recover"):
        result = r * scaling + torch.sum(q, dim=2, keepdim=True) * min

    with Timer("scale"):
        result = result / 2

    with Timer("softmax"):
        prob = sm(result)

    # with Timer("decompress v"):
    #     v = (quant_v * scaling + min).to(torch.float16)

    with Timer("v convert_to_fp16"):
        quant_v_ = quant_v.to(torch.float16)
    with Timer("quant v matmul"):
        r = torch.bmm(prob, quant_v_)

    with Timer("v recover"):
        result = r * scaling + torch.sum(prob, dim=2, keepdim=True) * min

    print(result.shape)

    # with Timer("transpose"):
    #     v.transpose(1, 2)

    # with Timer("v"):
    #     result = torch.bmm(result, v)
