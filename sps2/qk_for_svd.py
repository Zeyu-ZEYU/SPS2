#! /usr/bin/env python3


import pickle

import torch

with open("../../qk_for_svd.pkl", "rb") as f:
    qk_list = pickle.load(f)


svd_usvh_v_list = [[None for _ in range(16)] for _ in range(24)]

layer = 12
head = 10
tsr = qk_list[layer][head]
tsr = torch.concat(tsr, dim=0)

one = tsr[:512, :]
U, S, RT = torch.linalg.svd(one, full_matrices=False)
R = RT.transpose(0, 1)

for i in range(7):
    sec = tsr[512 * (i + 1) : 512 * (i + 2), :]
    U, S, RT = torch.linalg.svd(sec, full_matrices=False)
    R2 = RT.transpose(0, 1)
    diff = R2 - R
    # diffj = torch.abs(diff)
    # Rj = torch.abs(R)
    print(R2)
    # print(R)
    # print(ratio)


# for layer in range(24):
#     for head in range(16):
#         tsr = qk_list[layer][head]
#         tsr = torch.concat(tsr, dim=0)
#         U, S, V = torch.linalg.svd(tsr, full_matrices=False)
#         svd_usvh_v_list[layer][head] = V.transpose(0, 1)


# with open("./svd_usvh_v.pkl", "wb") as f:
#     pickle.dump(svd_usvh_v_list, f)
