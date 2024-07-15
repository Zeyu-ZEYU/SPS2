import torch
import time

device = torch.device("cuda:0")
b = torch.rand(5120,int(5120*3*0.3),dtype=torch.float16,device=device)
a = torch.rand(128, 5120,dtype=torch.float16,device=device)


torch.cuda.synchronize()
time1 = time.time()
for layer in range(40):
    torch.matmul(a,b)
torch.cuda.synchronize()
time2 = time.time()


diff = time2-time1
print(diff)
