"""
explainability.py
==================
SHAP-based explainability for the Payroll Anomaly Detection project.

Replaces global feature importance with SHAP (SHapley Additive exPlanations),
which provides mathematically grounded PER-RECORD explanations of model decisions.

Outputs:
    - reports/figures/shap_summary.png   → global feature impact across all records
    - reports/figures/shap_waterfall.png → explanation for one specific anomaly

Author: Lyna Medina Gassouma
Project: Intelligent Anomaly Detection in Global Payroll Data — ADP
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

PROCESSED_DIR = "../data/processed"
MODELS_DIR    = "../models"
FIGURES_DIR   = "../reports/figures"

FEATURE_COLS = ["country", "currency", "role", "department",
                "years_experience", "base_salary", "bonus",
                "tax", "social_security", "net_pay"]

os.makedirs(FIGURES_DIR, exist_ok=True)


def get_class1_shap(shap_values, expected_value):
    """
    Normalizes SHAP output across SHAP library versions.

    Different SHAP versions return shap_values either as:
        - a list [array_class0, array_class1]
        - a 3D array of shape (n_samples, n_features, n_classes)
        - a 2D array (already class 1 for binary models)

    Args:
        shap_values    : Raw output from explainer.shap_values()
        expected_value : Raw output from explainer.expected_value

    Returns:
        tuple: (shap_values_class1 [2D array], base_value_class1 [float])
    """
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    if isinstance(expected_value, (list, np.ndarray)):
        base = expected_value[1] if len(np.shape(expected_value)) > 0 and len(expected_value) > 1 else expected_value[0]
    else:
        base = expected_value

    return sv, base


def main():
    print("Loading model and data...")
    with open(os.path.join(MODELS_DIR, "random_forest.pkl"), "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(os.path.join(PROCESSED_DIR, "payroll_processed.csv"))

    # Sample for global summary plot (SHAP is slow on full 500k rows)
    sample = df.sample(100, random_state=42)
    X_sample = sample[FEATURE_COLS].values

    print("Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    sv, base_value = get_class1_shap(shap_values, explainer.expected_value)

    # ─── Global Summary Plot ───────────────────────────────────────────────
    print("Plotting global SHAP summary...")
    plt.figure(figsize=(9, 6))
    shap.summary_plot(sv, X_sample, feature_names=FEATURE_COLS, show=False)
    plt.title("SHAP Summary — Feature Impact on Anomaly Score", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path1 = os.path.join(FIGURES_DIR, "shap_summary.png")
    plt.savefig(path1, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path1}")

    # ─── Per-Record Waterfall Plot (one anomaly example) ──────────────────
    print("Plotting per-record SHAP waterfall...")
    anomalies = df[df["is_anomaly"] == 1]
    record = anomalies.iloc[0]
    X_record = record[FEATURE_COLS].values.reshape(1, -1)

    record_shap = explainer.shap_values(X_record)
    sv_record, base_record = get_class1_shap(record_shap, explainer.expected_value)

    plt.figure(figsize=(9, 6))
    shap.plots._waterfall.waterfall_legacy(
        base_record,
        sv_record[0],
        feature_names=FEATURE_COLS,
        max_display=10,
        show=False
    )
    plt.title(f"SHAP Explanation — Anomaly Record (type: {record['anomaly_type']})",
            fontsize=12, fontweight="bold")
    plt.tight_layout()
    path2 = os.path.join(FIGURES_DIR, "shap_waterfall.png")
    plt.savefig(path2, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path2}")

    print("\nDone. Both SHAP figures saved to reports/figures/")


if __name__ == "__main__":
    main()