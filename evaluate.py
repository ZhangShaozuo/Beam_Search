import pickle
with open("decode.pkl", "rb") as f:
    output = pickle.load(f)
print(output)