import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)