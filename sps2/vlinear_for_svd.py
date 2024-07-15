#! /usr/bin/env python3


import pickle

import torch

with open("./v_for_svd.pkl", "rb") as f:
    v_for_svd = pickle.load(f)
with open("./linear_for_svd.pkl", "rb") as f:
    linear_for_svd = pickle.load(f)


svd_usvh_v_for_vlinear = [[None for _ in range(16)] for _ in range(24)]


for layer in range(24):
    for head in range(16):
        v = v_for_svd[layer][head][0]
        linear = linear_for_svd[layer][head][0].t()
        tsr = torch.concat([v, linear], dim=0)
        U, S, V = torch.linalg.svd(tsr, full_matrices=False)
        svd_usvh_v_for_vlinear[layer][head] = V.transpose(0, 1)


with open("./svd_usvh_v_for_vlinear.pkl", "wb") as f:
    pickle.dump(svd_usvh_v_for_vlinear, f)
