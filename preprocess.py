"""
preprocess.py
-------------
Handles all data loading, cleaning, feature scaling, and SMOTE oversampling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV dataset and return a cleaned DataFrame."""
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded {len(df)} rows, {df.shape[1]} columns.")

    # Drop exact duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[INFO] Dropped {before - len(df)} duplicate rows.")

    # Drop rows with nulls
    df.dropna(inplace=True)

    return df


def preprocess(df: pd.DataFrame, fit_scaler: bool = True):
    """
    Scale Amount and Time. V1-V28 are already PCA-transformed.

    Parameters
    ----------
    df          : Raw DataFrame with columns Time, V1-V28, Amount, Class
    fit_scaler  : True during training (fit + save scaler), False during inference (load scaler)

    Returns
    -------
    X : numpy array of features
    y : numpy array of labels (if Class column present), else None
    """
    df = df.copy()

    scaler = StandardScaler()

    if fit_scaler:
        df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
        df['scaled_time']   = scaler.fit_transform(df[['Time']])
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"[INFO] Scaler saved to {SCALER_PATH}")
    else:
        scaler = joblib.load(SCALER_PATH)
        df['scaled_amount'] = scaler.transform(df[['Amount']])
        df['scaled_time']   = scaler.transform(df[['Time']])

    df.drop(['Amount', 'Time'], axis=1, inplace=True)

    if 'Class' in df.columns:
        X = df.drop('Class', axis=1).values
        y = df['Class'].values
        return X, y
    else:
        return df.values, None


def apply_smote(X: np.ndarray, y: np.ndarray):
    """
    Apply SMOTE to handle class imbalance.
    Returns balanced X_res, y_res.
    """
    print(f"[INFO] Before SMOTE — Fraud: {sum(y==1)}, Legit: {sum(y==0)}")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"[INFO] After  SMOTE — Fraud: {sum(y_res==1)}, Legit: {sum(y_res==0)}")
    return X_res, y_res


def get_train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
    """Split data into train and test sets, stratified."""
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else '../data/creditcard_sample.csv'
    df = load_data(path)
    X, y = preprocess(df, fit_scaler=True)
    X_res, y_res = apply_smote(X, y)
    print(f"[INFO] Final dataset shape: X={X_res.shape}, y={y_res.shape}")
