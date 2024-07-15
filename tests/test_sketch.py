import pickle

with open("../../large_files/gpt_params.pkl", "rb") as f:
    obj = pickle.load(f)

for a, b in obj:
    print(a)
