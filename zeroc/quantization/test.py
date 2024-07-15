import torch

device = torch.device("cuda")
t1 = torch.tensor([1, 2, 3, 4], device=device, dtype=torch.uint8)
t2 = torch.tensor([5, 6, 7, 8], device=device, dtype=torch.uint8)

t1 = t1 << 4
new = t1 + t2
new
t1 = new >> 4
t2 = new << 4 >> 4
print(t1, t2)
input = "How big is the universe? " * 500
print(len(input))
