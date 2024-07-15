import time

import torch
import torch.distributed as dist

dist.init_process_group("nccl", init_method=f"tcp://192.168.0.3:55634", rank=0, world_size=2)


tensor = torch.tensor([3, 3, 3], dtype=torch.float16, device="cuda:0")
recv = torch.empty_like(tensor, dtype=torch.float16, device="cuda:0")

if dist.get_rank() == 0:
    dist.send(tensor, dst=1)
    dist.recv(recv, src=1)
else:
    dist.recv(recv, src=0)
    dist.send(tensor, dst=0)


time.sleep(3)
