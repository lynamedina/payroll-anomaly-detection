import requests

payload = {
    "employee_id": "EMP_NORMAL_TEST",
    "country": "US",
    "currency": "USD",
    "role": "Payroll Specialist",
    "department": "Operations",
    "years_experience": 11.3,
    "base_salary": 73417.46,
    "bonus": 4143.60,
    "tax": 21871.07,
    "social_security": 7723.71,
    "net_pay": 47966.28
}

r = requests.post("http://127.0.0.1:8000/predict", json=payload)
print(r.json())