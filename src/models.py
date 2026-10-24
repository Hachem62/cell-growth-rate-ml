from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

POLY_DEGREES = [1, 2, 3, 4, 5, 6]
KNN_K_VALUES = [1, 3, 5, 10, 20, 50]
KNN_WEIGHTS = ["uniform", "distance"]


def make_polynomial_model(degree: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("regressor", LinearRegression()),
    ])


def make_knn_model(k: int, weights: str) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", KNeighborsRegressor(n_neighbors=k, weights=weights)),
    ])


def get_candidate_models() -> dict[str, dict]:
    candidates = {}
    for d in POLY_DEGREES:
        candidates[f"Polynomial (degree={d})"] = {
            "family": "Polynomial", "degree": d, "k": None, "weights": None,
            "model": make_polynomial_model(d),
        }
    for k in KNN_K_VALUES:
        for w in KNN_WEIGHTS:
            candidates[f"kNN (k={k}, {w})"] = {
                "family": "kNN", "degree": None, "k": k, "weights": w,
                "model": make_knn_model(k, w),
            }
    return candidates