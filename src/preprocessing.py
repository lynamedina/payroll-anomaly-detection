"""
preprocessing.py
================
Data pipeline for the Payroll Anomaly Detection project.

This module handles loading, cleaning, and preprocessing of all 4 datasets:
    1. synthetic_global_payroll.csv  — Main training dataset (labeled anomalies)
    2. ds_salaries.csv               — Salary benchmarks by role and country
    3. HRDataset_v14.csv             — Real employee structure for EDA enrichment
    4. creditcard.csv                — Labeled fraud dataset for model benchmarking

Output:
    - data/processed/payroll_processed.csv       → ready for model training
    - data/processed/salaries_processed.csv      → salary benchmarks
    - data/processed/hr_processed.csv            → HR enrichment data
    - data/processed/creditcard_processed.csv    → fraud benchmark data

Author: [Lyna Medina Gassouma]
Project: Intelligent Anomaly Detection in Global Payroll Data — ADP
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────

RAW_DIR  = "../data"
OUT_DIR  = "../data/processed"

PATHS = {
    "payroll":    os.path.join(RAW_DIR, "synthetic_global_payroll.csv"),
    "salaries":   os.path.join(RAW_DIR, "ds_salaries.csv"),
    "hr":         os.path.join(RAW_DIR, "HRDataset_v14.csv"),
    "creditcard": os.path.join(RAW_DIR, "creditcard.csv"),
}

os.makedirs(OUT_DIR, exist_ok=True)


# ─── Utility ──────────────────────────────────────────────────────────────────

def report(df: pd.DataFrame, name: str) -> None:
    """
    Prints a quick summary of a DataFrame after processing.

    Args:
        df   (pd.DataFrame): The processed dataframe.
        name (str)         : Dataset label for display.
    """
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Rows      : {len(df):,}")
    print(f"  Columns   : {len(df.columns)}")
    print(f"  Nulls     : {df.isnull().sum().sum()}")
    print(f"  Dtypes    : {dict(df.dtypes.value_counts())}")
    print(f"  Columns   : {list(df.columns)}")


# ─── 1. Payroll Dataset ───────────────────────────────────────────────────────

def preprocess_payroll(path: str = PATHS["payroll"]) -> pd.DataFrame:
    """
    Loads and preprocesses the synthetic global payroll dataset.

    This is the MAIN training dataset for all anomaly detection models.

    Cleaning steps:
        - Drop rows with null values
        - Remove rows where base_salary or net_pay <= 0
        - Remove duplicate employee records
        - Normalize numerical features with MinMaxScaler
        - Encode categorical features (country, currency, role) with LabelEncoder
        - Keep is_anomaly and anomaly_type as labels (not normalized)

    Features after processing:
        Numerical  : base_salary, bonus, tax, social_security, net_pay (all normalized)
        Categorical: country, currency, role (label encoded)
        Labels     : is_anomaly (0/1), anomaly_type (string)

    Args:
        path (str): Path to synthetic_global_payroll.csv

    Returns:
        pd.DataFrame: Cleaned and normalized payroll dataframe ready for
                      model training (Random Forest, Gradient Boosting,
                      ANN, Isolation Forest).
    """
    print(f"\n[1/4] Loading payroll dataset from: {path}")
    df = pd.read_csv(path)

    # ── Basic cleaning
    initial_len = len(df)
    df.dropna(inplace=True)
    df.drop_duplicates(subset=["employee_id"], inplace=True)

    # Remove rows with non-positive salaries (data generation artifacts)
    df = df[df["base_salary"] > 0]
    df = df[df["net_pay"] > 0]
    print(f"  Removed {initial_len - len(df)} invalid/duplicate rows")

    # ── Encode categorical columns
    cat_cols = ["country", "currency", "role", "department"]
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # ── Normalize numerical columns (exclude labels and IDs)
    num_cols = ["base_salary", "bonus", "tax", "social_security", "net_pay"]
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # ── Save
    out_path = os.path.join(OUT_DIR, "payroll_processed.csv")
    df.to_csv(out_path, index=False)
    report(df, "Payroll Dataset (Main Training Data)")
    print(f"  Anomalies : {df['is_anomaly'].sum()} / {len(df)} rows")
    print(f"  Saved to  : {out_path}")

    return df


# ─── 2. DS Salaries Dataset ───────────────────────────────────────────────────

def preprocess_salaries(path: str = PATHS["salaries"]) -> pd.DataFrame:
    """
    Loads and preprocesses the ds_salaries dataset.

    Used as a salary benchmark reference — NOT for model training.
    Helps validate that salary ranges in the synthetic payroll data
    are realistic for each role and country.

    Cleaning steps:
        - Drop nulls and duplicates
        - Keep only relevant columns: job_title, salary_in_usd, company_location,
          experience_level, employment_type
        - Filter out extreme outliers (salary > $1M or < $10k)
        - Compute benchmark stats: mean, std, min, max per job_title

    Args:
        path (str): Path to ds_salaries.csv

    Returns:
        pd.DataFrame: Cleaned salary benchmark dataframe with stats per role.
    """
    print(f"\n[2/4] Loading salaries dataset from: {path}")
    df = pd.read_csv(path)

    # ── Keep relevant columns
    keep = ["job_title", "salary_in_usd", "company_location",
            "experience_level", "employment_type"]
    df = df[[c for c in keep if c in df.columns]]

    # ── Clean
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # ── Remove extreme outliers
    df = df[(df["salary_in_usd"] > 10000) & (df["salary_in_usd"] < 1_000_000)]

    # ── Compute benchmark stats per job title
    benchmarks = df.groupby("job_title")["salary_in_usd"].agg(
        mean_salary="mean",
        std_salary="std",
        min_salary="min",
        max_salary="max",
        count="count"
    ).reset_index()
    benchmarks["std_salary"].fillna(0, inplace=True)

    # ── Save
    out_path = os.path.join(OUT_DIR, "salaries_processed.csv")
    benchmarks.to_csv(out_path, index=False)
    report(benchmarks, "Salaries Dataset (Benchmark Reference)")
    print(f"  Unique roles : {benchmarks['job_title'].nunique()}")
    print(f"  Saved to     : {out_path}")

    return benchmarks


# ─── 3. HR Dataset ────────────────────────────────────────────────────────────

def preprocess_hr(path: str = PATHS["hr"]) -> pd.DataFrame:
    """
    Loads and preprocesses the HRDataset_v14 dataset.

    Used for EDA enrichment only — provides real employee structure
    (department, tenure, performance, salary) to contextualize findings.
    NOT used for model training.

    Cleaning steps:
        - Drop nulls and duplicates
        - Keep only payroll-relevant columns
        - Normalize salary column
        - Encode categorical columns (Department, Position, etc.)

    Args:
        path (str): Path to HRDataset_v14.csv

    Returns:
        pd.DataFrame: Cleaned HR dataframe for EDA use.
    """
    print(f"\n[3/4] Loading HR dataset from: {path}")
    df = pd.read_csv(path)

    # ── Keep payroll-relevant columns
    keep = ["Employee_Name", "Department", "Position", "Salary",
            "ManagerName", "EmpStatusID", "PerfScoreID",
            "EngagementSurvey", "EmpSatisfaction", "Termd"]
    df = df[[c for c in keep if c in df.columns]]

    # ── Clean
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # ── Remove invalid salaries
    if "Salary" in df.columns:
        df = df[df["Salary"] > 0]

    # ── Encode categoricals
    cat_cols = ["Department", "Position", "ManagerName"]
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # ── Normalize salary
    if "Salary" in df.columns:
        scaler = MinMaxScaler()
        df[["Salary"]] = scaler.fit_transform(df[["Salary"]])

    # ── Save
    out_path = os.path.join(OUT_DIR, "hr_processed.csv")
    df.to_csv(out_path, index=False)
    report(df, "HR Dataset (EDA Enrichment)")
    print(f"  Saved to : {out_path}")

    return df


# ─── 4. Credit Card Fraud Dataset ─────────────────────────────────────────────

def preprocess_creditcard(path: str = PATHS["creditcard"]) -> pd.DataFrame:
    """
    Loads and preprocesses the Credit Card Fraud Detection dataset.

    Used as a BENCHMARK dataset to validate anomaly detection model performance
    on a real labeled fraud dataset (not payroll-specific).

    This dataset already comes normalized (PCA features V1-V28).
    Only Amount needs scaling. Class (0=normal, 1=fraud) is the label.

    Cleaning steps:
        - Drop nulls and duplicates
        - Normalize the Amount column with MinMaxScaler
        - Drop the Time column (not relevant for model comparison)
        - Rename Class → is_anomaly for consistency with payroll dataset

    Args:
        path (str): Path to creditcard.csv

    Returns:
        pd.DataFrame: Cleaned fraud benchmark dataframe with is_anomaly label.
    """
    print(f"\n[4/4] Loading credit card dataset from: {path}")

    # creditcard.csv is large (~150MB) — load in chunks to avoid memory issues
    chunks = []
    for chunk in pd.read_csv(path, chunksize=50000):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    # ── Clean
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # ── Drop Time (irrelevant for cross-domain benchmarking)
    if "Time" in df.columns:
        df.drop(columns=["Time"], inplace=True)

    # ── Normalize Amount
    if "Amount" in df.columns:
        scaler = MinMaxScaler()
        df[["Amount"]] = scaler.fit_transform(df[["Amount"]])

    # ── Rename Class → is_anomaly for consistency
    if "Class" in df.columns:
        df.rename(columns={"Class": "is_anomaly"}, inplace=True)

    # ── Save
    out_path = os.path.join(OUT_DIR, "creditcard_processed.csv")
    df.to_csv(out_path, index=False)
    report(df, "Credit Card Dataset (Fraud Benchmark)")
    print(f"  Fraud cases : {df['is_anomaly'].sum():,} / {len(df):,} rows")
    print(f"  Saved to    : {out_path}")

    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_all():
    """
    Runs the full preprocessing pipeline for all 4 datasets.

    Execution order:
        1. Payroll   → main training data
        2. Salaries  → benchmark reference
        3. HR        → EDA enrichment
        4. Creditcard→ fraud benchmark

    All outputs are saved to data/processed/
    """
    print("\n" + "="*50)
    print("  PAYROLL ANOMALY DETECTION — Preprocessing Pipeline")
    print("="*50)

    payroll    = preprocess_payroll()
    salaries   = preprocess_salaries()
    hr         = preprocess_hr()
    creditcard = preprocess_creditcard()

    print("\n" + "="*50)
    print("  Pipeline complete. All files saved to data/processed/")
    print("="*50)

    return payroll, salaries, hr, creditcard


if __name__ == "__main__":
    run_all()
