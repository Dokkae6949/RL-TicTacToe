import pickle

with open("qtable.pkl", "rb") as f:
    qtable = pickle.load(f)

print(qtable)
print(len(qtable))
