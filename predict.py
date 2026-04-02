"""
predict.py
----------
Fraud probability prediction functions.
Loads a trained model and returns fraud probability + decision.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
RF_PATH    = os.path.join(MODELS_DIR, 'random_forest.pkl')
LR_PATH    = os.path.join(MODELS_DIR, 'logistic_regression.pkl')

# Fraud decision threshold (tune to balance precision/recall)
FRAUD_THRESHOLD = 0.50


def load_model(model_type: str = 'random_forest'):
    """
    Load a saved model from disk.

    Parameters
    ----------
    model_type : 'random_forest' or 'logistic_regression'
    """
    path = RF_PATH if model_type == 'random_forest' else LR_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. "
            "Please run src/train.py first to train and save the model."
        )
    return joblib.load(path)


def predict_single(transaction: dict, model_type: str = 'random_forest') -> dict:
    """
    Predict fraud probability for a single transaction dict.

    Parameters
    ----------
    transaction : dict with keys Time, V1-V28, Amount
    model_type  : 'random_forest' or 'logistic_regression'

    Returns
    -------
    dict with keys:
        fraud_probability (float 0-1)
        is_fraud          (bool)
        risk_level        (str: LOW / MEDIUM / HIGH)
        model_used        (str)
    """
    model = load_model(model_type)

    df = pd.DataFrame([transaction])
    X, _ = preprocess(df, fit_scaler=False)

    fraud_prob = float(model.predict_proba(X)[0][1])
    is_fraud   = fraud_prob >= FRAUD_THRESHOLD

    if fraud_prob >= 0.80:
        risk_level = "HIGH"
    elif fraud_prob >= 0.50:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        'fraud_probability': round(fraud_prob, 4),
        'is_fraud':          is_fraud,
        'risk_level':        risk_level,
        'model_used':        model_type,
        'threshold_used':    FRAUD_THRESHOLD,
    }


def predict_batch(df: pd.DataFrame, model_type: str = 'random_forest') -> pd.DataFrame:
    """
    Predict fraud for all rows in a DataFrame.

    Parameters
    ----------
    df         : DataFrame with columns Time, V1-V28, Amount
    model_type : 'random_forest' or 'logistic_regression'

    Returns
    -------
    Original DataFrame with extra columns:
        fraud_probability, is_fraud, risk_level
    """
    model  = load_model(model_type)
    X, _   = preprocess(df.copy(), fit_scaler=False)
    probas = model.predict_proba(X)[:, 1]

    result = df.copy()
    result['fraud_probability'] = probas.round(4)
    result['is_fraud']          = probas >= FRAUD_THRESHOLD
    result['risk_level']        = pd.cut(
        probas,
        bins=[-0.001, 0.499, 0.799, 1.001],
        labels=['LOW', 'MEDIUM', 'HIGH']
    )
    return result


if __name__ == '__main__':
    # Quick smoke test with a dummy transaction
    sample = {
        'Time': 100,
        'V1': -1.36, 'V2': -0.07, 'V3': 2.54,  'V4': 1.38,
        'V5': -0.34, 'V6': 0.46,  'V7': 0.24,  'V8': 0.10,
        'V9': 0.36,  'V10': 0.09, 'V11': -0.55,'V12': -0.62,
        'V13': -0.99,'V14': -0.31,'V15': 1.47, 'V16': -0.47,
        'V17': 0.21, 'V18': 0.03, 'V19': 0.40, 'V20': 0.25,
        'V21': -0.02,'V22': 0.28, 'V23': -0.11,'V24': 0.07,
        'V25': 0.13, 'V26': -0.19,'V27': 0.13, 'V28': -0.02,
        'Amount': 149.62
    }
    result = predict_single(sample)
    print("\n--- Prediction Result ---")
    for k, v in result.items():
        print(f"  {k}: {v}")
