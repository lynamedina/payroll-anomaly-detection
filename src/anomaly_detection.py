"""
anomaly_detection.py
====================
Model training and evaluation for the Payroll Anomaly Detection project.

This module trains and evaluates 4 anomaly detection models on the preprocessed
payroll dataset, then benchmarks them against the credit card fraud dataset.

Models:
    1. Random Forest        — Supervised, ensemble tree-based classifier
    2. Gradient Boosting    — Supervised, boosted ensemble classifier
    3. ANN                  — Supervised, Artificial Neural Network (MLP)
    4. Isolation Forest     — Unsupervised, anomaly-score based detector

Inputs:
    - data/processed/payroll_processed.csv      → main training/test data
    - data/processed/creditcard_processed.csv   → cross-domain benchmark

Outputs:
    - models/random_forest.pkl
    - models/gradient_boosting.pkl
    - models/ann.pkl
    - models/isolation_forest.pkl
    - data/processed/payroll_predictions.csv    → predictions + scores per row

Author: [Lyna Medina Gassouma]
Project: Intelligent Anomaly Detection in Global Payroll Data — ADP
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score
)

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────

PROCESSED_DIR = "../data/processed"
MODELS_DIR    = "../models"

PAYROLL_PATH     = os.path.join(PROCESSED_DIR, "payroll_processed.csv")
CREDITCARD_PATH  = os.path.join(PROCESSED_DIR, "creditcard_processed.csv")

os.makedirs(MODELS_DIR, exist_ok=True)


# ─── Utility ──────────────────────────────────────────────────────────────────

def save_model(model, filename: str) -> None:
    """
    Serializes and saves a trained model to the models/ directory using pickle.

    Args:
        model    : Trained sklearn model object.
        filename (str): Output filename (e.g. 'random_forest.pkl').
    """
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved → {path}")


def print_metrics(name: str, y_true, y_pred, y_score=None) -> dict:
    """
    Prints and returns evaluation metrics for a trained model.

    Metrics computed:
        - Precision  : ratio of true anomalies among flagged rows
        - Recall     : ratio of actual anomalies correctly detected
        - F1-Score   : harmonic mean of precision and recall
        - ROC-AUC    : area under the ROC curve (if scores provided)
        - Confusion Matrix

    Args:
        name    (str)        : Model name for display.
        y_true  (array-like) : Ground truth labels (0/1).
        y_pred  (array-like) : Predicted labels (0/1).
        y_score (array-like) : Anomaly scores or probabilities (optional).

    Returns:
        dict: Dictionary of metric names and values.
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    auc       = roc_auc_score(y_true, y_score) if y_score is not None else None

    print(f"\n{'─'*50}")
    print(f"  {name} — Results")
    print(f"{'─'*50}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    if auc:
        print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {confusion_matrix(y_true, y_pred)}")
    print(f"\n  Full Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))

    return {"model": name, "precision": precision, "recall": recall,
            "f1": f1, "auc": auc}


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_payroll_data(path: str = PAYROLL_PATH):
    """
    Loads the preprocessed payroll dataset, splits into train/test sets,
    and applies SMOTE to the training set to fix class imbalance.

    SMOTE (Synthetic Minority Oversampling Technique) generates synthetic
    anomaly records to balance the 95.6% normal / 4.4% anomaly ratio.
    Applied ONLY on training data — test set stays imbalanced (realistic).

    Returns:
        tuple: (X_train_resampled, X_test, y_train_resampled, y_test, df, feature_cols)
    """
    from imblearn.over_sampling import SMOTE

    print(f"\nLoading payroll data from: {path}")
    df = pd.read_csv(path)

    feature_cols = ["country", "currency", "role", "department",
                "years_experience", "base_salary", "bonus",
                "tax", "social_security", "net_pay",
                "tax_rate", "net_to_gross_ratio", "bonus_rate",
                "ss_rate", "total_deduction_rate"]
    X = df[feature_cols].values
    y = df["is_anomaly"].values

    # Split BEFORE resampling — test set must stay original
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Before SMOTE — Train: {len(X_train):,} rows | "
        f"Anomaly rate: {y_train.mean():.2%}")

    # Apply SMOTE only on training data
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print(f"  After SMOTE  — Train: {len(X_train):,} rows | "
        f"Anomaly rate: {y_train.mean():.2%}")
    print(f"  Test set     — {len(X_test):,} rows | "
        f"Anomaly rate: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test, df, feature_cols

# ─── Model 1: Random Forest ───────────────────────────────────────────────────

def train_random_forest(X_train, X_test, y_train, y_test) -> tuple:
    """
    Trains a Random Forest classifier for supervised payroll anomaly detection.

    Random Forest builds multiple decision trees on random subsets of features
    and aggregates their votes. Well-suited for tabular payroll data with mixed
    feature types (categorical encoded + normalized numerical).

    Hyperparameters:
        n_estimators = 100   : number of trees
        max_depth    = None  : trees grow until pure (handles complex patterns)
        class_weight = balanced : compensates for the 5% anomaly imbalance
        random_state = 42

    Args:
        X_train (ndarray): Training features.
        X_test  (ndarray): Test features.
        y_train (ndarray): Training labels.
        y_test  (ndarray): Test labels.

    Returns:
        tuple: (trained model, predictions, anomaly probabilities)
    """
    print("\n[1/4] Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]

    # Find threshold that maximizes F1 score
    from sklearn.metrics import precision_recall_curve, f1_score
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_score)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    print(f"  Optimal threshold: {optimal_threshold:.4f} (default was 0.5)")
    y_pred = (y_score >= optimal_threshold).astype(int)

    metrics = print_metrics("Random Forest", y_test, y_pred, y_score)
    save_model(model, "random_forest.pkl")

    return model, y_pred, y_score, metrics


# ─── Model 2: Gradient Boosting ───────────────────────────────────────────────

def train_gradient_boosting(X_train, X_test, y_train, y_test) -> tuple:
    """
    Trains a Gradient Boosting classifier for supervised payroll anomaly detection.

    Gradient Boosting builds trees sequentially, each correcting the errors of
    the previous one. Typically achieves higher precision than Random Forest
    on imbalanced datasets when tuned correctly.

    Hyperparameters:
        n_estimators  = 200  : number of boosting stages
        learning_rate = 0.05 : shrinkage — lower = more robust, slower
        max_depth     = 4    : shallow trees reduce overfitting
        subsample     = 0.8  : stochastic gradient boosting (80% row sampling)
        random_state  = 42

    Args:
        X_train (ndarray): Training features.
        X_test  (ndarray): Test features.
        y_train (ndarray): Training labels.
        y_test  (ndarray): Test labels.

    Returns:
        tuple: (trained model, predictions, anomaly probabilities, metrics dict)
    """
    print("\n[2/4] Training Gradient Boosting...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]

    # Find threshold that maximizes F1 score
    from sklearn.metrics import precision_recall_curve, f1_score
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_score)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    print(f"  Optimal threshold: {optimal_threshold:.4f} (default was 0.5)")
    y_pred = (y_score >= optimal_threshold).astype(int)

    metrics = print_metrics("Gradient Boosting", y_test, y_pred, y_score)
    save_model(model, "gradient_boosting.pkl")

    return model, y_pred, y_score, metrics


# ─── Model 3: ANN (MLP) ───────────────────────────────────────────────────────

def train_ann(X_train, X_test, y_train, y_test) -> tuple:
    """
    Trains an Artificial Neural Network (MLP) for supervised payroll anomaly detection.

    Uses sklearn's MLPClassifier — a fully connected feedforward neural network.
    Architecture: Input(8) → Dense(64, ReLU) → Dense(32, ReLU) → Output(1, Sigmoid)

    The network learns non-linear relationships between payroll features
    (e.g. net_pay being higher than base_salary) that rule-based systems miss.

    Hyperparameters:
        hidden_layer_sizes = (64, 32) : two hidden layers
        activation         = relu
        solver             = adam     : adaptive learning rate optimizer
        max_iter           = 300      : training epochs
        early_stopping     = True     : stops if validation loss stops improving
        random_state       = 42

    Args:
        X_train (ndarray): Training features.
        X_test  (ndarray): Test features.
        y_train (ndarray): Training labels.
        y_test  (ndarray): Test labels.

    Returns:
        tuple: (trained model, predictions, anomaly probabilities, metrics dict)
    """
    print("\n[3/4] Training ANN (MLP)...")
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]

    # Find threshold that maximizes F1 score
    from sklearn.metrics import precision_recall_curve, f1_score
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_score)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    print(f"  Optimal threshold: {optimal_threshold:.4f} (default was 0.5)")
    y_pred = (y_score >= optimal_threshold).astype(int)

    metrics = print_metrics("ANN (MLP)", y_test, y_pred, y_score)
    save_model(model, "ann.pkl")

    return model, y_pred, y_score, metrics


# ─── Model 4: Isolation Forest ────────────────────────────────────────────────

def train_isolation_forest(X_train, X_test, y_test) -> tuple:
    """
    Trains an Isolation Forest for UNSUPERVISED payroll anomaly detection.

    Unlike the other 3 models, Isolation Forest does NOT use labels during training.
    It detects anomalies by isolating observations using random splits —
    anomalies are isolated faster (fewer splits needed) than normal points.

    This is the most realistic production scenario: in real payroll systems,
    you rarely have labeled anomaly data, so unsupervised detection is critical.

    Hyperparameters:
        n_estimators      = 200  : number of isolation trees
        contamination     = 0.05 : expected anomaly rate (matches our 5% injection)
        max_samples       = auto : uses min(256, n_samples)
        random_state      = 42

    Note on output mapping:
        Isolation Forest outputs: -1 (anomaly), 1 (normal)
        We remap to: 1 (anomaly), 0 (normal) for consistency with other models.

    Args:
        X_train (ndarray): Training features (labels NOT used).
        X_test  (ndarray): Test features.
        y_test  (ndarray): True labels (used for evaluation only).

    Returns:
        tuple: (trained model, predictions, anomaly scores, metrics dict)
    """
    print("\n[4/4] Training Isolation Forest (Unsupervised)...")
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    # Train WITHOUT labels — purely unsupervised
    model.fit(X_train)

    # Predict: -1 = anomaly, 1 = normal → remap to 1/0
    raw_pred = model.predict(X_test)
    y_pred   = np.where(raw_pred == -1, 1, 0)

    # Anomaly score: negative = more anomalous → invert for probability-like score
    y_score  = -model.score_samples(X_test)
    # Normalize scores to [0, 1] for comparability
    y_score  = (y_score - y_score.min()) / (y_score.max() - y_score.min())

    metrics = print_metrics("Isolation Forest", y_test, y_pred, y_score)
    save_model(model, "isolation_forest.pkl")

    return model, y_pred, y_score, metrics

# ─── Model 5: XGBoost ────────────────────────────────────────────────

def train_xgboost(X_train, X_test, y_train, y_test) -> tuple:
    """
    Trains an XGBoost classifier for supervised payroll anomaly detection.

    XGBoost (eXtreme Gradient Boosting) is the industry-standard algorithm
    for tabular data in Big Data / production ML systems. It typically
    outperforms standard Gradient Boosting through regularization,
    optimized tree-splitting, and built-in handling of feature interactions.

    Hyperparameters:
        n_estimators     = 200  : boosting rounds
        learning_rate    = 0.05 : shrinkage
        max_depth        = 5    : tree depth
        subsample        = 0.8  : row sampling per tree
        colsample_bytree = 0.8  : feature sampling per tree

    Args:
        X_train, X_test, y_train, y_test: train/test splits.

    Returns:
        tuple: (trained model, predictions, scores, metrics)
    """
    print("\n[5/6] Training XGBoost...")
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]

    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_score)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    y_pred = (y_score >= optimal_threshold).astype(int)

    metrics = print_metrics("XGBoost", y_test, y_pred, y_score)
    save_model(model, "xgboost.pkl")
    return model, y_pred, y_score, metrics

# ─── Model 6: autoencoder ────────────────────────────────────────────────

def train_autoencoder(X_train, X_test, y_train, y_test) -> tuple:
    """
    Trains a Deep Learning Autoencoder for unsupervised payroll anomaly detection.

    An autoencoder is a neural network trained to RECONSTRUCT its input.
    It is trained ONLY on normal (non-anomalous) records — it learns to
    compress and reconstruct "normal payroll patterns" through a bottleneck.

    At inference time, anomalous records produce HIGH reconstruction error
    (the network has never seen patterns like them), while normal records
    reconstruct with LOW error. This error becomes the anomaly score.

    Architecture: Input(15) -> Dense(8) -> Dense(4, bottleneck) -> Dense(8) -> Output(15)

    Why this matters for the project:
        Unlike Isolation Forest, the autoencoder learns a dense, continuous
        representation of "normal payroll behavior" — a genuinely deep
        learning approach to unsupervised anomaly detection, standard in
        production fraud/anomaly systems (e.g. credit card fraud, network intrusion).

    Args:
        X_train, X_test, y_train, y_test: train/test splits (labels used
                                        ONLY to filter training to normal
                                        records and for evaluation).

    Returns:
        tuple: (trained model, predictions, reconstruction-error scores, metrics)
    """
    print("\n[6/6] Training Autoencoder (Deep Learning, Unsupervised)...")

    # Scale features specifically for the neural network (mixed-scale inputs)
    scaler = StandardScaler()
    X_train_normal = X_train[y_train == 0]
    X_train_scaled = scaler.fit_transform(X_train_normal)
    X_test_scaled  = scaler.transform(X_test)

    n_features = X_train.shape[1]

    autoencoder = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(8, activation="relu"),
        layers.Dense(4, activation="relu"),   # bottleneck
        layers.Dense(8, activation="relu"),
        layers.Dense(n_features, activation="linear"),
    ])
    autoencoder.compile(optimizer="adam", loss="mse")

    autoencoder.fit(
        X_train_scaled, X_train_scaled,
        epochs=20, batch_size=256,
        validation_split=0.1, verbose=0
    )

    # Reconstruction error = anomaly score
    reconstructions = autoencoder.predict(X_test_scaled, verbose=0)
    mse = np.mean(np.square(X_test_scaled - reconstructions), axis=1)
    y_score = (mse - mse.min()) / (mse.max() - mse.min())

    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_score)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    y_pred = (y_score >= optimal_threshold).astype(int)

    metrics = print_metrics("Autoencoder", y_test, y_pred, y_score)

    autoencoder.save("../models/autoencoder.h5")
    print("  Model saved → ../models/autoencoder.h5")

    return autoencoder, y_pred, y_score, metrics

# ─── Save Predictions ─────────────────────────────────────────────────────────

def save_predictions(df: pd.DataFrame, feature_cols: list,
                    rf_pred, gb_pred, ann_pred, if_pred,
                    rf_score, gb_score, ann_score, if_score) -> None:
    """
    Appends all model predictions and anomaly scores to the payroll dataframe
    and saves it for use in the dashboard/API.

    Output columns added:
        rf_pred, gb_pred, ann_pred, if_pred     : binary predictions (0/1)
        rf_score, gb_score, ann_score, if_score : anomaly scores [0, 1]
        ensemble_score                           : average of all 4 scores
        ensemble_flag                            : 1 if ensemble_score > 0.5

    Args:
        df           (pd.DataFrame) : Full payroll dataframe.
        feature_cols (list)         : Feature column names used for splitting.
        rf_pred, gb_pred, ann_pred, if_pred   : Predictions from each model.
        rf_score, gb_score, ann_score, if_score: Scores from each model.
    """
    # Only annotate the test portion (last 20%)
    test_size = len(rf_pred)
    df_out = df.copy().tail(test_size).reset_index(drop=True)

    df_out["rf_pred"]   = rf_pred
    df_out["gb_pred"]   = gb_pred
    df_out["ann_pred"]  = ann_pred
    df_out["if_pred"]   = if_pred

    df_out["rf_score"]  = rf_score
    df_out["gb_score"]  = gb_score
    df_out["ann_score"] = ann_score
    df_out["if_score"]  = if_score

    # Ensemble: average score across all 4 models
    df_out["ensemble_score"] = (rf_score + gb_score + ann_score + if_score) / 4
    df_out["ensemble_flag"]  = (df_out["ensemble_score"] > 0.5).astype(int)

    out_path = os.path.join(PROCESSED_DIR, "payroll_predictions.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\n  Predictions saved → {out_path}")
    print(f"  Ensemble flagged  : {df_out['ensemble_flag'].sum()} anomalies")


# ─── Comparison Summary ───────────────────────────────────────────────────────

def print_summary(results: list) -> None:
    """
    Prints a side-by-side comparison table of all model metrics.

    Args:
        results (list): List of metric dicts returned by print_metrics().
    """
    print(f"\n{'='*60}")
    print(f"  MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<25} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print(f"  {'─'*55}")
    for r in results:
        auc = f"{r['auc']:.4f}" if r['auc'] else "  N/A  "
        print(f"  {r['model']:<25} {r['precision']:>10.4f} {r['recall']:>8.4f} "
            f"{r['f1']:>8.4f} {auc:>8}")
    print(f"{'='*60}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_all():
    """
    Runs the full model training and evaluation pipeline.

    Steps:
        1. Load preprocessed payroll data
        2. Train Random Forest (supervised)
        3. Train Gradient Boosting (supervised)
        4. Train ANN / MLP (supervised)
        5. Train Isolation Forest (unsupervised)
        6. Save all models as .pkl files
        7. Save predictions + ensemble scores to CSV
        8. Print comparison summary
    """
    print("\n" + "="*60)
    print("  PAYROLL ANOMALY DETECTION — Model Training Pipeline")
    print("="*60)

    # Load data
    X_train, X_test, y_train, y_test, df, feature_cols = load_payroll_data()

    # Train models
    rf_model,  rf_pred,  rf_score,  rf_metrics  = train_random_forest(X_train, X_test, y_train, y_test)
    gb_model,  gb_pred,  gb_score,  gb_metrics  = train_gradient_boosting(X_train, X_test, y_train, y_test)
    ann_model, ann_pred, ann_score, ann_metrics  = train_ann(X_train, X_test, y_train, y_test)
    if_model,  if_pred,  if_score,  if_metrics  = train_isolation_forest(X_train, X_test, y_test)
    xgb_model, xgb_pred, xgb_score, xgb_metrics = train_xgboost(X_train, X_test, y_train, y_test)
    ae_model,  ae_pred,  ae_score,  ae_metrics  = train_autoencoder(X_train, X_test, y_train, y_test)

    # Save predictions
    save_predictions(df, feature_cols,
                    rf_pred, gb_pred, ann_pred, if_pred,
                    rf_score, gb_score, ann_score, if_score)

    # Summary
    print_summary([rf_metrics, gb_metrics, ann_metrics, if_metrics, xgb_metrics, ae_metrics])


if __name__ == "__main__":
    run_all()
