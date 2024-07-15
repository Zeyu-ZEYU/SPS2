import time

import torch
import zc_blas
import zc_bmm_half

# a = torch.tensor(
#     [
#         [[1, 2, 3], [1, 2, 3]],
#         [[1, 2, 3], [9, 2, 3]],
#         [[7, 2, 3], [1, 2, 3]],
#         [[1, 2, 3], [1, 2, 3]],
#     ],
#     dtype=torch.float16,
#     device="cuda:0",
# )
# b = torch.tensor(
#     [
#         [[1, 2, 3], [1, 2, 3]],
#         [[1, 2, 3], [9, 2, 3]],
#         [[7, 2, 3], [1, 2, 3]],
#         [[1, 2, 3], [1, 2, 3]],
#     ],
#     dtype=torch.float16,
#     device="cuda:0",
# )
# b = torch.tensor(
#     [
#         [[1, 2], [3, 3], [1, 2]],
#         [[88, 2], [4, 6], [1, 2]],
#         [[8, 5], [1, 2], [1, 2]],
#         [[4, 2], [1, 2], [11, 2]],
#     ],
#     dtype=torch.float16,
#     device="cuda:0",
# )


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


a = torch.ones([4, 16 * 1024, 128], dtype=torch.uint8, device="cuda:0")
b = torch.ones([4, 16 * 1024, 128], dtype=torch.uint8, device="cuda:0")

# c = torch.ones([4, 8192, 128], dtype=torch.half, device="cuda:0")
# d = torch.ones([4, 128, 8192], dtype=torch.half, device="cuda:0")

# for _ in range(10):
#     with _Timer("test2"):
#         torch.bmm(c, d)
for _ in range(10):
    with _Timer("test"):
        zc_blas.bmm_uint8(a, b)
        # zc_blas_uint8.bmm_uint8(a, b)
        # zc_blas.bmm_half(a, b)


# print(zc_blas.bmm_half(a, b))
