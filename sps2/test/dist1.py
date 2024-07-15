#! /usr/bin/env python3


import time

import torch
import torch.distributed as dist

dist.init_process_group("nccl", init_method="tcp://128.143.69.201:39223", rank=1, world_size=2)
torch.cuda.set_device("cuda:1")
device = torch.device("cuda:1")

# with torch.cuda.stream(torch.cuda.Stream()):
for i in range(1):
    dist.isend(torch.tensor(i, device=device, dtype=torch.int64), dst=0)
for i in range(1):
    tensor2 = torch.tensor(1, device=device, dtype=torch.int64)
    dist.irecv(tensor2, src=0)
    print(tensor2)

time.sleep(2222)
