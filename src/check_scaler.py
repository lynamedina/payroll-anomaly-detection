# save as src/check_scaler.py
import pickle
import numpy as np

with open("../models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print("Scaler data_min_ (should be raw salary values like 20000, not 0):")
print(scaler.data_min_)
print("\nScaler data_max_ (should be raw salary values like 500000, not 1):")
print(scaler.data_max_)