# save as src/debug_api.py
import pickle
import numpy as np

with open("../models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("../models/random_forest.pkl", "rb") as f:
    model = pickle.load(f)

# Exact same values as get_normal.py — DE, Intern, HR
COUNTRY_MAP  = {"AU":0,"BR":1,"CA":2,"DE":3,"FR":4,"IN":5,"JP":6,"TN":7,"UK":8,"US":9}
CURRENCY_MAP = {"AUD":0,"BRL":1,"CAD":2,"EUR":3,"GBP":4,"INR":5,"JPY":6,"TND":7,"USD":8}
ROLE_MAP     = {"Data Scientist":0,"DevOps Engineer":1,"Director":2,"Finance Analyst":3,
                "HR Manager":4,"Intern":5,"Junior Engineer":6,"Payroll Specialist":7,
                "Project Manager":8,"Sales Manager":9,"Senior Engineer":10,"Support Specialist":11}
DEPT_MAP     = {"Engineering":0,"Finance":1,"HR":2,"IT":3,"Operations":4,"Sales":5}

base=34260.71; tax=8900.9; ss=3286.27; bon=801.8; net=22875.34

features = np.array([[
    COUNTRY_MAP["DE"],
    CURRENCY_MAP["EUR"],
    ROLE_MAP["Intern"],
    DEPT_MAP["HR"],
    13.8,
    base, bon, tax, ss, net,
    tax/base, net/base, bon/base, ss/base, (tax+ss)/base
]], dtype=float)

print("Raw features:")
print(features)
print("\nScaled features:")
scaled = scaler.transform(features)
print(scaled)
print(f"\nPrediction: {model.predict(scaled)[0]}")
print(f"Score: {model.predict_proba(scaled)[0][1]:.4f}")