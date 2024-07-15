#! /usr/bin/env python3


import pickle

import torch

with open("/u/qxc4fh/zeyu_workspace/v_for_svd.pkl", "rb") as f:
    v_for_svd = pickle.load(f)
with open("/u/qxc4fh/zeyu_workspace/linear_for_svd.pkl", "rb") as f:
    linear_for_svd = pickle.load(f)

svd_usvh_v_list = [[None for _ in range(16)] for _ in range(24)]

str_list = []

for layer in range(24):
    for head in range(16):
        v = v_for_svd[layer][head][0]
        linear = linear_for_svd[layer][head][0].t()
        tsr = torch.concat([v, linear], dim=0)
        U, S, V = torch.linalg.svd(tsr, full_matrices=False)
        l = [x.item() for x in S]
        str_list.append(l)
        # svd_usvh_v_for_vlinear[layer][head] = V.transpose(0, 1)


str_list.sort(key=lambda x: x[0], reverse=True)
wstr = ""
for each in str_list:
    line = ""
    for x in each:
        line += f"{x:.4f},"
    line += "\n"
    wstr += line


with open("./vlinear_singular_values.csv", "w") as f:
    f.write(wstr)
