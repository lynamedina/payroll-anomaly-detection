import pandas as pd
import numpy as np

np.random.seed(42)

N = 500000
ANOMALY_RATE = 0.05

countries = ["US", "FR", "UK", "DE", "TN", "IN", "CA", "AU", "BR", "JP"]
currencies = {
    "US": "USD", "FR": "EUR", "UK": "GBP", "DE": "EUR",
    "TN": "TND", "IN": "INR", "CA": "CAD", "AU": "AUD",
    "BR": "BRL", "JP": "JPY"
}
roles = {
    "Junior Engineer":     (40000,  70000),
    "Senior Engineer":     (80000,  130000),
    "Data Scientist":      (85000,  125000),
    "Payroll Specialist":  (45000,  75000),
    "HR Manager":          (65000,  100000),
    "Director":            (120000, 190000),
    "Intern":              (20000,  35000),
    "Project Manager":     (75000,  120000),
    "Finance Analyst":     (60000,  95000),
    "DevOps Engineer":     (85000,  135000),
    "Support Specialist":  (35000,  60000),
    "Sales Manager":       (70000,  115000),
}
departments = ["Engineering", "Finance", "HR", "Sales", "Operations", "IT"]

records = []
for i in range(N):
    role = np.random.choice(list(roles.keys()))
    country = np.random.choice(countries)
    low, high = roles[role]
    base = round(np.random.uniform(low, high), 2)
    tax = round(base * np.random.uniform(0.15, 0.30), 2)
    social_sec = round(base * np.random.uniform(0.06, 0.12), 2)
    bonus = round(base * np.random.uniform(0.0, 0.15), 2)
    net = round(base - tax - social_sec + bonus, 2)

    is_anomaly = 0
    anomaly_type = "none"

    if np.random.random() < ANOMALY_RATE:
        is_anomaly = 1
        anomaly_type = np.random.choice([
            "salary_spike",
            "missing_tax",
            "bonus_exceeded",
            "net_exceeds_gross",
            "zero_deductions",
            "duplicate_payment",
            "currency_mismatch",
            "ghost_employee",
        ])
        if anomaly_type == "salary_spike":
            base = round(base * np.random.uniform(3, 6), 2)
        elif anomaly_type == "missing_tax":
            tax = 0.0
        elif anomaly_type == "bonus_exceeded":
            bonus = round(base * np.random.uniform(1.5, 3), 2)
        elif anomaly_type == "net_exceeds_gross":
            net = round(base * np.random.uniform(1.2, 1.8), 2)
        elif anomaly_type == "zero_deductions":
            tax = 0.0
            social_sec = 0.0
        elif anomaly_type == "duplicate_payment":
            # Same employee paid twice — net_pay doubled
            net = round(net * 2, 2)
        elif anomaly_type == "currency_mismatch":
            # Wrong currency applied (e.g. TND salary treated as USD)
            base = round(base * 3.2, 2)  # TND to USD conversion error
        elif anomaly_type == "ghost_employee":
            # Employee with no role history getting paid
            base = round(np.random.uniform(1000, 5000), 2)
            net = base
        net = round(base - tax - social_sec + bonus, 2)

    records.append({
        "employee_id":    f"EMP{i+1:05d}",
        "country":        country,
        "currency":       currencies[country],
        "role":           role,
        "base_salary":    base,
        "bonus":          bonus,
        "tax":            tax,
        "social_security": social_sec,
        "net_pay":        net,
        "is_anomaly":     is_anomaly,
        "anomaly_type":   anomaly_type,
        "department":     np.random.choice(departments),
        "years_experience": round(np.random.uniform(0, 30), 1),
    })

df = pd.DataFrame(records)
df.to_csv("../data/synthetic_global_payroll.csv", index=False)
print(df["anomaly_type"].value_counts())
print(f"\n{df['is_anomaly'].sum()} anomalies out of {len(df)} rows")