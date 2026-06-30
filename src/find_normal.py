# save as src/find_normal.py
import pickle, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

with open('../models/random_forest.pkl', 'rb') as f:
    model = pickle.load(f)
with open('../models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

raw = pd.read_csv('../data/synthetic_global_payroll.csv')
raw = raw[raw['is_anomaly'] == 0].head(1000)

feature_cols = ['country','currency','role','department','years_experience',
                'base_salary','bonus','tax','social_security','net_pay']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in ['country','currency','role','department']:
    raw[col] = le.fit_transform(raw[col].astype(str))

raw['tax_rate']             = raw['tax'] / raw['base_salary']
raw['net_to_gross_ratio']   = raw['net_pay'] / raw['base_salary']
raw['bonus_rate']           = raw['bonus'] / raw['base_salary']
raw['ss_rate']              = raw['social_security'] / raw['base_salary']
raw['total_deduction_rate'] = (raw['tax'] + raw['social_security']) / raw['base_salary']

all_cols = ['country','currency','role','department','years_experience',
            'base_salary','bonus','tax','social_security','net_pay',
            'tax_rate','net_to_gross_ratio','bonus_rate','ss_rate','total_deduction_rate']

X = scaler.transform(raw[all_cols].values)
scores = model.predict_proba(X)[:, 1]

# Find records with lowest scores
raw_orig = pd.read_csv('../data/synthetic_global_payroll.csv')
raw_orig = raw_orig[raw_orig['is_anomaly'] == 0].head(1000)
raw_orig['score'] = scores
best = raw_orig.nsmallest(3, 'score')[['country','currency','role','department',
       'years_experience','base_salary','bonus','tax','social_security','net_pay','score']]
print(best.to_string())