import os, sys

path = os.path.join("Artifacts", "train.csv")

os.makedirs(os.path.dirname(path), exist_ok= True)

print(os.path.dirname(path))