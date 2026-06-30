# save as src/check_encoding.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

raw_df = pd.read_csv("../data/synthetic_global_payroll.csv")
le = LabelEncoder()

for col in ["country", "currency", "role", "department"]:
    le.fit(raw_df[col].astype(str))
    classes = list(le.classes_)
    print(f"\n{col}:")
    for i, c in enumerate(classes):
        print(f"  {i}: {c}")