"""
train.py
--------
Trains Logistic Regression and Random Forest models on the fraud dataset.
Saves trained models to the /models directory.
"""

import os
import sys
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_data, preprocess, apply_smote, get_train_test_split

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

LR_PATH = os.path.join(MODELS_DIR, 'logistic_regression.pkl')
RF_PATH = os.path.join(MODELS_DIR, 'random_forest.pkl')


def evaluate_model(name: str, model, X_test: np.ndarray, y_test: np.ndarray):
    """Print classification report and AUC-ROC for a fitted model."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))
    print(f"AUC-ROC : {roc_auc_score(y_test, y_proba):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    return y_proba


def plot_roc_curves(y_test, lr_proba, rf_proba, save_path=None):
    """Plot and optionally save ROC curves for both models."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, proba in [('Logistic Regression', lr_proba), ('Random Forest', rf_proba)]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', lw=2)

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] ROC curve saved to {save_path}")
    plt.show()


def plot_feature_importance(rf_model, feature_names, save_path=None):
    """Plot top-15 feature importances from the Random Forest."""
    importances = rf_model.feature_importances_
    indices     = np.argsort(importances)[::-1][:15]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(15), importances[indices], color='steelblue', edgecolor='white')
    ax.set_xticks(range(15))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.set_title('Top 15 Feature Importances (Random Forest)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Importance Score')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] Feature importance plot saved to {save_path}")
    plt.show()


def train(data_path: str):
    # ── 1. Load and preprocess ────────────────────────────────────
    df = load_data(data_path)
    X, y = preprocess(df, fit_scaler=True)

    # ── 2. SMOTE ─────────────────────────────────────────────────
    X_res, y_res = apply_smote(X, y)

    # ── 3. Train/test split ───────────────────────────────────────
    X_train, X_test, y_train, y_test = get_train_test_split(X_res, y_res)

    feature_names = [c for c in df.columns
                     if c not in ('Amount', 'Time', 'Class')] + ['scaled_amount', 'scaled_time']

    # ── 4. Logistic Regression ────────────────────────────────────
    print("\n[INFO] Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr.fit(X_train, y_train)
    joblib.dump(lr, LR_PATH)
    print(f"[INFO] Logistic Regression saved → {LR_PATH}")
    lr_proba = evaluate_model("Logistic Regression", lr, X_test, y_test)

    # ── 5. Random Forest ─────────────────────────────────────────
    print("\n[INFO] Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf, RF_PATH)
    print(f"[INFO] Random Forest saved → {RF_PATH}")
    rf_proba = evaluate_model("Random Forest", rf, X_test, y_test)

    # ── 6. Plots ─────────────────────────────────────────────────
    plot_roc_curves(y_test, lr_proba, rf_proba)
    plot_feature_importance(rf, feature_names)

    print("\n[INFO] Training complete.")
    return lr, rf


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else '../data/creditcard_sample.csv'
    train(path)
