import pandas as pd
import numpy as np
from pathlib import Path

COLUMNS = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX",
    "RM", "AGE", "DIS", "RAD", "TAX",
    "PTRATIO", "B", "LSTAT", "MEDV",
]

FEATURE_DESCRIPTIONS = {
    "CRIM":    "Per capita crime rate by town",
    "ZN":      "Proportion of residential land zoned for lots >25,000 sq.ft.",
    "INDUS":   "Proportion of non-retail business acres per town",
    "CHAS":    "Charles River dummy variable (1 if tract bounds river)",
    "NOX":     "Nitric oxides concentration (parts per 10 million)",
    "RM":      "Average number of rooms per dwelling",
    "AGE":     "Proportion of owner-occupied units built prior to 1940",
    "DIS":     "Weighted distances to five Boston employment centres",
    "RAD":     "Index of accessibility to radial highways",
    "TAX":     "Full-value property-tax rate per $10,000",
    "PTRATIO": "Pupil-teacher ratio by town",
    "B":       "1000(Bk - 0.63)^2, Bk = proportion of Black residents",
    "LSTAT":   "% lower status of the population",
    "MEDV":    "Median value of owner-occupied homes in $1,000s (TARGET)",
}


def load_data(path: str | Path = "housing.csv") -> pd.DataFrame:
    df = pd.read_csv(path, header=None, sep=r"\s+", names=COLUMNS)
    return df


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=["MEDV"])
    y = df["MEDV"]
    return X, y
