#! /usr/bin/env python3


import pickle

import torch

with open("./token_for_svd.pkl", "rb") as f:
    token_for_svd = pickle.load(f)
# with open("./linear_for_svd.pkl", "rb") as f:
#     linear_for_svd = pickle.load(f)


svd_usvh_v_for_token = [[None for _ in range(16)] for _ in range(24)]


for layer in range(24):
    for head in range(16):
        input = token_for_svd[layer][0]
        addon = token_for_svd[layer][head + 1].t()
        combo = torch.concat([input, addon], dim=0)
        U, S, V = torch.linalg.svd(combo, full_matrices=False)
        svd_usvh_v_for_token[layer][head] = V.transpose(0, 1)


with open("./svd_usvh_v_for_token.pkl", "wb") as f:
    pickle.dump(svd_usvh_v_for_token, f)
