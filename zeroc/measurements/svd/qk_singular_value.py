#! /usr/bin/env python3


import pickle

import torch

with open("/u/qxc4fh/zeyu_workspace/qk_for_svd.pkl", "rb") as f:
    qk_list = pickle.load(f)


svd_usvh_v_list = [[None for _ in range(16)] for _ in range(24)]

str_list = []

for layer in range(24):
    for head in range(16):
        tsr = qk_list[layer][head]
        tsr = torch.concat(tsr, dim=0)
        U, S, V = torch.linalg.svd(tsr, full_matrices=False)
        l = [x.item() for x in S]
        str_list.append(l)
wstr = ""
for each in str_list:
    line = ""
    for x in each:
        line += f"{x:.4f},"
    line += "\n"
    wstr += line

with open("./qk_singular_values.csv", "w") as f:
    f.write(wstr)
