#import pandas as pd
#df = pd.read_csv('../data/synthetic_global_payroll.csv')
#normal = df[df['is_anomaly']==0].iloc[0]
#print(normal[['country','role','department','years_experience','base_salary','bonus','tax','social_security','net_pay']])


#.. import pickle, numpy as np
#with open('../models/scaler.pkl', 'rb') as f:
#    scaler = pickle.load(f)
#with open('../models/random_forest.pkl', 'rb') as f:
#    model = pickle.load(f)


#import pickle, numpy as np
#with open('../models/scaler.pkl', 'rb') as f:
#    scaler = pickle.load(f)
#with open('../models/random_forest.pkl', 'rb') as f:
#    model = pickle.load(f)

#features = np.array([[3, 3, 9, 2, 13.8, 34260.71, 801.8, 8900.9, 3286.27, 22875.34,
#                    8900.9/34260.71, 22875.34/34260.71, 801.8/34260.71,
#                    3286.27/34260.71, (8900.9+3286.27)/34260.71]])
#features_scaled = scaler.transform(features)
#pred = model.predict(features_scaled)[0]
#score = model.predict_proba(features_scaled)[0][1]
#print(f'Prediction: {pred} | Score: {score:.4f}')

"""
import requests

# Real normal record from dataset
payload = {
    "employee_id": "EMP_NORMAL_001",
    "country": "DE",
    "currency": "EUR",
    "role": "Intern",
    "department": "HR",
    "years_experience": 13.8,
    "base_salary": 34260.71,
    "bonus": 801.80,
    "tax": 8900.90,
    "social_security": 3286.27,
    "net_pay": 22875.34
}

r = requests.post("http://127.0.0.1:8000/predict", json=payload)
result = r.json()
print(f"Prediction : {'ANOMALY' if result['is_anomaly'] else 'NORMAL'}")
print(f"Score      : {result['anomaly_score']}")
print(f"Risk Level : {result['risk_level']}")
print(f"Message    : {result['message']}")

import pickle
with open('../models/thresholds.pkl', 'rb') as f:
    t = pickle.load(f)
print(t)
"""

import pickle, numpy as np, pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix

# Load model and test data
with open('../models/random_forest.pkl', 'rb') as f:
    model = pickle.load(f)
with open('../models/thresholds.pkl', 'rb') as f:
    thresh = pickle.load(f)

df = pd.read_csv('../data/processed/payroll_processed.csv')
feature_cols = ['country','currency','role','department','years_experience',
                'base_salary','bonus','tax','social_security','net_pay',
                'tax_rate','net_to_gross_ratio','bonus_rate','ss_rate','total_deduction_rate']
from sklearn.model_selection import train_test_split
X = df[feature_cols].values
y = df['is_anomaly'].values
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

with open('../models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
X_test_scaled = scaler.transform(X_test)

score = model.predict_proba(X_test_scaled)[:, 1]
pred = (score >= thresh['random_forest']).astype(int)
print('RF Precision:', precision_score(y_test, pred))
print('RF Recall:', recall_score(y_test, pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, pred))