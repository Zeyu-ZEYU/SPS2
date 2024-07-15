import time

import numpy as np
import torch

# # 创建两个整数类型的矩阵
# a = torch.tensor([[1, 2], [3, 4]], dtype=torch.int8)
# b = torch.tensor([[5, 6], [7, 8]], dtype=torch.int8)

# # 矩阵乘法
# c = torch.matmul(a, b)

# print(c)


# exit()


# def uint64(num):
#     return torch.tensor(num, dtype=torch.uint16, device="cuda:0")


# x1, x2, x3, x4 = (uint64(127), uint64(33), uint64(4), uint64(11))
# y1, y2, y3, y4 = uint64((173, 7, 21, 22))

# print(y2)


# X13 = (x1 << 24) | (x3 << 8)
# X24 = (x2 << 16) | x4

# Y13 = (y1 << 24) | (y3 << 8)
# Y24 = (y2 << 16) | y4


# print(Y13)


# a = torch.randint(1, 10, [2, 8192, 128], dtype=torch.int32, device="cuda:0")
# b = torch.randint(1, 10, [2, 128, 8192], dtype=torch.int32, device="cuda:0")


# torch.cuda.synchronize()
# t1 = time.time()
# for _ in range(20):
#     torch.bmm(a, b)
# torch.cuda.synchronize()
# t2 = time.time()

# print(1000 * (t2 - t1))

dtype = np.uint32
x1 = np.array([2], dtype=dtype)
x2 = np.array([4], dtype=dtype)
x3 = np.array([3], dtype=dtype)
x4 = np.array([3], dtype=dtype)

x13 = (x1 << 4) | x3
x24 = (x2 << 4) | x4


# X13 = (x1 << 24) | (x3 << 8)
# X24 = (x2 << 16) | x4

# Y13 = (y1 << 24) | (y3 << 8)
# Y24 = (y2 << 16) | y4
m = x13 * x24

print(np.binary_repr(x13[0], width=32))
print(np.binary_repr(x24[0], width=32))
print(np.binary_repr(m[0], width=32))

# mask1 = np.array([0xFFFF0000FFFF0000], dtype=dtype)
# mask2 = np.array([0x0000FFFF0000FFFF], dtype=dtype)
# res = ((X13 * Y13) & mask1) | (X24 * Y24 & mask2)

# print(x1 * y1)
# print(res >> 48)
# print(x2 * y2)
# print(res << 16 >> 48)
# print(x3 * y3)
# print(res << 32 >> 48)
# print(x4 * y4)
# print(res << 48 >> 48)


# f = np.array((127, 33, 4, 11), dtype=np.uint64)
# x1, x2, x3, x4 = f << 4
# y1, y2, y3, y4 = np.array((173, 7, 21, 22), dtype=np.uint64)
# # example

# print(np.binary_repr(x13y13[0], width=64))
# print(np.binary_repr(mask1[0], width=64))
# print(np.binary_repr(mask1_x13y13[0], width=64))

# X13 = (x1 << 24) | (x3 << 8)
# X24 = (x2 << 16) | x4

# Y13 = (y1 << 24) | (y3 << 8)
# Y24 = (y2 << 16) | y4
