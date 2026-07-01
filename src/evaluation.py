"""
evaluation.py
=============
Model evaluation and visualization for the Payroll Anomaly Detection project.

This module loads all 4 trained models, runs evaluation on the test set,
and generates all figures needed for the technical report and notebook.

Outputs (saved to reports/figures/):
    - roc_curves.png          → ROC curves for all 4 models
    - confusion_matrices.png  → Confusion matrices side by side
    - feature_importance.png  → Random Forest feature importance
    - anomaly_breakdown.png   → Anomaly type distribution
    - score_distribution.png  → Anomaly score distributions
    - ensemble_analysis.png   → Ensemble score vs true labels

Author: [Lyna Medina Gassouma]
Project: Intelligent Anomaly Detection in Global Payroll Data — ADP
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    roc_curve, auc,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────

PROCESSED_DIR = "../data/processed"
MODELS_DIR    = "../models"
FIGURES_DIR   = "../reports/figures"
RAW_DIR       = "../data"

os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── Style ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi":        150,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "font.family":       "sans-serif",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})

COLORS = {
    "Random Forest":     "#2196F3",
    "Gradient Boosting": "#4CAF50",
    "ANN (MLP)":         "#FF9800",
    "Isolation Forest":  "#9C27B0",
    "anomaly":           "#F44336",
    "normal":            "#2196F3",
}

MODEL_NAMES = ["Random Forest", "Gradient Boosting", "ANN (MLP)", "Isolation Forest"]
MODEL_FILES = ["random_forest.pkl", "gradient_boosting.pkl", "ann.pkl", "isolation_forest.pkl"]

FEATURE_COLS = [
    "country", "currency", "role", "department", "years_experience",
    "base_salary", "bonus", "tax", "social_security", "net_pay",
    "tax_rate", "net_to_gross_ratio", "bonus_rate", "ss_rate", "total_deduction_rate"
]


# ─── Load Data & Models ───────────────────────────────────────────────────────

def load_data():
    """
    Loads the preprocessed payroll dataset and splits into train/test sets.
    Uses the same random_state=42 and test_size=0.2 as anomaly_detection.py
    to ensure identical splits.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, df_raw)
    """
    print("[1/7] Loading data...")
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "payroll_processed.csv"))
    df_raw = pd.read_csv(os.path.join(RAW_DIR, "synthetic_global_payroll.csv"))

    X = df[FEATURE_COLS].values
    y = df["is_anomaly"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Test set: {len(X_test):,} rows | {y_test.sum():,} anomalies")
    return X_train, X_test, y_train, y_test, df_raw


def load_models():
    """
    Loads all 4 trained models from the models/ directory.

    Returns:
        dict: {model_name: model_object}
    """
    print("[2/7] Loading models...")
    models = {}
    for name, fname in zip(MODEL_NAMES, MODEL_FILES):
        path = os.path.join(MODELS_DIR, fname)
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
        print(f"  Loaded: {name}")
    return models


def get_predictions(models: dict, X_test, y_test):
    """
    Runs all models on the test set and collects predictions and scores.

    For Isolation Forest:
        - Remaps -1/1 output to 1/0 (anomaly/normal)
        - Normalizes score_samples to [0,1]

    For supervised models:
        - Uses predict() for labels
        - Uses predict_proba()[:,1] for scores

    Args:
        models (dict)    : Loaded model objects.
        X_test (ndarray) : Test features.
        y_test (ndarray) : True labels.

    Returns:
        dict: {model_name: {"pred": array, "score": array}}
    """
    # Load optimal thresholds
    import pickle
    try:
        with open(os.path.join(MODELS_DIR, "thresholds.pkl"), "rb") as f:
            thresholds = pickle.load(f)
    except:
        thresholds = {}
        print("  Warning: thresholds.pkl not found, using 0.5")

    results = {}
    for name, model in models.items():
        if name == "Isolation Forest":
            raw   = model.predict(X_test)
            pred  = np.where(raw == -1, 1, 0)
            score = -model.score_samples(X_test)
            score = (score - score.min()) / (score.max() - score.min())
        else:
            score = model.predict_proba(X_test)[:, 1]
            # Map display name to thresholds.pkl key
            name_map = {
                "Random Forest":     "random_forest",
                "Gradient Boosting": "gradient_boosting",
                "ANN (MLP)":         "ann",
                "XGBoost":           "xgboost",
            }
            thresh_key = name_map.get(name, name.lower().replace(" ", "_"))
            thresh = thresholds.get(thresh_key, 0.5)
            print(f"  {name}: using threshold {thresh:.4f}")
            pred = (score >= thresh).astype(int)

        results[name] = {"pred": pred, "score": score}
        print(f"  {name}: {pred.sum():,} flagged")
    return results


# ─── Plot 1: ROC Curves ───────────────────────────────────────────────────────

def plot_roc_curves(results: dict, y_test):
    """
    Plots ROC curves for all 4 models on a single axes.

    ROC curve shows the tradeoff between True Positive Rate (Recall)
    and False Positive Rate at various classification thresholds.
    AUC (Area Under Curve) summarizes performance in a single number:
        1.0 = perfect | 0.5 = random

    Saved to: reports/figures/roc_curves.png

    Args:
        results (dict)   : Predictions and scores per model.
        y_test  (ndarray): True labels.
    """
    print("[4/7] Plotting ROC curves...")
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["score"])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr,
                label=f"{name} (AUC = {roc_auc:.4f})",
                color=COLORS[name], linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curves — All Models on Payroll Test Set")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "roc_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


# ─── Plot 2: Confusion Matrices ───────────────────────────────────────────────

def plot_confusion_matrices(results: dict, y_test):
    """
    Plots confusion matrices for all 4 models in a 2x2 grid.

    Matrix layout:
        TN | FP
        FN | TP

    TN = correctly identified normal rows
    FP = normal rows wrongly flagged (false alarms)
    FN = missed anomalies
    TP = correctly caught anomalies

    Saved to: reports/figures/confusion_matrices.png

    Args:
        results (dict)   : Predictions per model.
        y_test  (ndarray): True labels.
    """
    print("[5/7] Plotting confusion matrices...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res["pred"])
        sns.heatmap(
            cm, annot=True, fmt=",", cmap="Blues",
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"],
            ax=axes[i], cbar=False,
            annot_kws={"size": 12}
        )
        axes[i].set_title(f"{name}", fontweight="bold")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

        # Annotate TN/FP/FN/TP
        tn, fp, fn, tp = cm.ravel()
        axes[i].set_xlabel(
            f"Predicted  |  TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}"
        )

    fig.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ─── Plot 3: Feature Importance ───────────────────────────────────────────────

def plot_feature_importance(models: dict):
    """
    Plots feature importance from the Random Forest model.

    Feature importance = mean decrease in impurity across all trees.
    Higher value = feature contributes more to anomaly detection decisions.

    This plot answers: "Which payroll fields are most indicative of anomalies?"

    Saved to: reports/figures/feature_importance.png

    Args:
        models (dict): Loaded model objects (uses Random Forest).
    """
    print("[6/7] Plotting feature importance...")
    rf = models["Random Forest"]
    importances = rf.feature_importances_
    indices     = np.argsort(importances)[::-1]
    sorted_features = [FEATURE_COLS[i] for i in indices]
    sorted_values   = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(sorted_features[::-1], sorted_values[::-1],
                   color=COLORS["Random Forest"], alpha=0.8)

    # Add value labels
    for bar, val in zip(bars, sorted_values[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8)

    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)")
    ax.set_title("Feature Importance — Random Forest\n"
                 "Which payroll fields best detect anomalies?")
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "feature_importance.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


# ─── Plot 4: Anomaly Type Breakdown ───────────────────────────────────────────

def plot_anomaly_breakdown(df_raw: pd.DataFrame):
    """
    Plots the distribution of injected anomaly types in the dataset.

    Shows both count and percentage for each anomaly type, helping
    contextualize model performance per category.

    Saved to: reports/figures/anomaly_breakdown.png

    Args:
        df_raw (pd.DataFrame): Raw (non-normalized) payroll dataframe
                               containing the anomaly_type column.
    """
    print("[7/7] Plotting anomaly breakdown...")
    anomalies = df_raw[df_raw["anomaly_type"] != "none"]["anomaly_type"]
    counts    = anomalies.value_counts()
    pcts      = (counts / len(anomalies) * 100).round(1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    colors = plt.cm.Set2(np.linspace(0, 1, len(counts)))
    bars = axes[0].bar(counts.index, counts.values, color=colors, alpha=0.85)
    axes[0].set_title("Anomaly Type Distribution — Count")
    axes[0].set_xlabel("Anomaly Type")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                     f"{val:,}", ha="center", fontsize=8)

    # Pie chart
    axes[1].pie(counts.values, labels=counts.index,
                autopct="%1.1f%%", colors=colors,
                startangle=140, pctdistance=0.8)
    axes[1].set_title("Anomaly Type Distribution — Percentage")

    fig.suptitle(f"Injected Anomaly Types\n"
                 f"Total: {len(anomalies):,} anomalies out of {len(df_raw):,} rows "
                 f"({len(anomalies)/len(df_raw)*100:.1f}%)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "anomaly_breakdown.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


# ─── Plot 5: Score Distribution ───────────────────────────────────────────────

def plot_score_distributions(results: dict, y_test):
    """
    Plots the distribution of anomaly scores for normal vs anomalous rows.

    A good model should show clear separation between the two distributions.
    Overlap indicates uncertainty at the decision boundary.

    Saved to: reports/figures/score_distribution.png

    Args:
        results (dict)   : Scores per model.
        y_test  (ndarray): True labels.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (name, res) in enumerate(results.items()):
        scores  = res["score"]
        normal  = scores[y_test == 0]
        anomaly = scores[y_test == 1]

        axes[i].hist(normal,  bins=60, alpha=0.6,
                     color=COLORS["normal"],  label="Normal",  density=True)
        axes[i].hist(anomaly, bins=60, alpha=0.6,
                     color=COLORS["anomaly"], label="Anomaly", density=True)
        axes[i].set_title(f"{name} — Score Distribution")
        axes[i].set_xlabel("Anomaly Score")
        axes[i].set_ylabel("Density")
        axes[i].legend()

    fig.suptitle("Anomaly Score Distributions\n"
                 "Separation between Normal and Anomalous rows per model",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "score_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


# ─── Print Final Metrics Table ────────────────────────────────────────────────

def print_final_table(results: dict, y_test):
    """
    Prints a clean comparison table of all model metrics.
    Copy this output directly into your LaTeX report.

    Args:
        results (dict)   : Predictions and scores per model.
        y_test  (ndarray): True labels.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    print(f"\n{'='*65}")
    print(f"  FINAL MODEL COMPARISON — Copy to Report")
    print(f"{'='*65}")
    print(f"  {'Model':<25} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print(f"  {'─'*60}")

    for name, res in results.items():
        p   = precision_score(y_test, res["pred"], zero_division=0)
        r   = recall_score(y_test, res["pred"], zero_division=0)
        f1  = f1_score(y_test, res["pred"], zero_division=0)
        auc_ = roc_auc_score(y_test, res["score"])
        print(f"  {name:<25} {p:>10.4f} {r:>8.4f} {f1:>8.4f} {auc_:>8.4f}")

    print(f"{'='*65}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_all():
    """
    Runs the full evaluation pipeline.

    Steps:
        1. Load preprocessed data and split (same seed as training)
        2. Load all 4 trained models
        3. Generate predictions and scores
        4. Plot ROC curves
        5. Plot confusion matrices
        6. Plot feature importance (Random Forest)
        7. Plot anomaly type breakdown
        8. Plot score distributions
        9. Print final comparison table
    """
    print("\n" + "="*60)
    print("  PAYROLL ANOMALY DETECTION — Evaluation Pipeline")
    print("="*60)

    X_train, X_test, y_train, y_test, df_raw = load_data()
    models  = load_models()
    results = get_predictions(models, X_test, y_test)

    plot_roc_curves(results, y_test)
    plot_confusion_matrices(results, y_test)
    plot_feature_importance(models)
    plot_anomaly_breakdown(df_raw)
    plot_score_distributions(results, y_test)
    print_final_table(results, y_test)

    print(f"All figures saved to: {FIGURES_DIR}")
    print("Done.\n")


if __name__ == "__main__":
    run_all()
