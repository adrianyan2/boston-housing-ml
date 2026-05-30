import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["RM2"]         = df["RM"] ** 2
    df["LOG_CRIM"]    = np.log1p(df["CRIM"])
    df["LOG_DIS"]     = np.log1p(df["DIS"])
    df["LOG_LSTAT"]   = np.log1p(df["LSTAT"])
    df["TAX_PTRATIO"] = df["TAX"] * df["PTRATIO"]
    df["AGE_DIS"]     = df["AGE"] / (df["DIS"] + 1e-6)
    return df


def get_splits(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_scaler_pipeline(model):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  model),
    ])
