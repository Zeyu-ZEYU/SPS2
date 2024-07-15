import torch
import time

device = torch.device("cuda:0")
t = torch.rand(1024*1024*2,128,dtype=torch.float32,device=device)

torch.cuda.synchronize()
time1 = time.time()
torch.linalg.svd(t, full_matrices=False)
torch.cuda.synchronize()
time2 = time.time()

diff = time2-time1
print(diff)
