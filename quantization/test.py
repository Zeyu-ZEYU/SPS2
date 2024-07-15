# import torch

# table = torch.tensor([10, 39, 18, 23], dtype=torch.float, device="cuda:0")

# compressed = torch.tensor([[0, 1, 2, 1, 1, 1], [3, 3, 3, 3, 3, 1]], dtype=torch.uint8, device=table.device)

# a = table[compressed]

# print(a)

import torch

M = torch.tril(torch.ones((6, 6), device="cuda"))

print(hasattr(torch, "float8_e5m2"))
