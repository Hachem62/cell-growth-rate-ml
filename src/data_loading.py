from pathlib import Path

import pandas as pd

TARGET = "Specific Growth Rate [d^-1]"
FEATURES = ["Temperature [C]", "pH", "Glucose [mM]", "Lactate [mM]"]


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:

    df = pd.read_csv(path)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    df = df.dropna(subset=FEATURES + [TARGET])
    return df[FEATURES], df[TARGET]