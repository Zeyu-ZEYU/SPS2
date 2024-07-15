import time

import torch
import zc_bmm_half
import zc_bmm_uint8


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
        torch.cuda.synchronize()
        self.start = time.time()
        return self

    def __exit__(self, *args):
        torch.cuda.synchronize()
        interval = 1000 * (time.time() - self.start)
        self.total_time += interval
        print(f"{self.name}: {interval}ms out of {self.total_time}ms.")


a = torch.tensor(
    [
        [[1, 2, 3], [1, 2, 3]],
        [[1, 2, 3], [9, 2, 3]],
        [[7, 2, 3], [1, 2, 3]],
        [[1, 2, 3], [1, 2, 3]],
    ],
    dtype=torch.float16,
    device="cuda:0",
)
b = torch.tensor(
    [
        [[1, 2], [3, 3], [1, 2]],
        [[88, 2], [4, 6], [1, 2]],
        [[8, 5], [1, 2], [1, 2]],
        [[4, 2], [1, 2], [11, 2]],
    ],
    dtype=torch.float16,
    device="cuda:0",
)


a = torch.randint(
    -10,
    10,
    [4, 8192, 128],
    dtype=torch.float16,
    device="cuda:0",
)
b = torch.randint(
    -10,
    10,
    [4, 128, 8192],
    dtype=torch.float16,
    device="cuda:0",
)
c = torch.randint(
    -10,
    10,
    [4, 8192, 128],
    dtype=torch.float16,
    device="cuda:0",
)
d = torch.randint(
    -10,
    10,
    [4, 128, 8192],
    dtype=torch.float16,
    device="cuda:0",
)
int_a = torch.randint(
    -10,
    10,
    [4, 8192, 128],
    dtype=torch.float16,
    device="cuda:0",
).to(torch.uint8)
int_b = torch.randint(
    -10,
    10,
    [4, 128, 8192],
    dtype=torch.float16,
    device="cuda:0",
).to(torch.uint8)


with _Timer("fp16"):
    fp16 = zc_bmm_half.call(a, b)

with _Timer("fp16 2"):
    fp16 = zc_bmm_half.call(a, b)

with _Timer("torch fp16"):
    t_fp16 = torch.bmm(a, b)

with _Timer("torch fp16 2"):
    t_fp16 = torch.bmm(c, d)

with _Timer("torch fp16 3"):
    t_fp16 = torch.bmm(c, d)

with _Timer("uint8"):
    uint8 = zc_bmm_uint8.call(int_a, int_b)

print(fp16)
print(t_fp16)
print(fp16 - t_fp16)
