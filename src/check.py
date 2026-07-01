import pandas as pd
df = pd.read_csv('../data/processed/payroll_processed.csv')
feature_cols = ['country','currency','role','department','years_experience',
                'base_salary','bonus','tax','social_security','net_pay',
                'tax_rate','net_to_gross_ratio','bonus_rate','ss_rate','total_deduction_rate']
print('All 15 features present:', all(c in df.columns for c in feature_cols))