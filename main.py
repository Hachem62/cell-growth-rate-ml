from pathlib import Path

import pandas as pd

from src.data_loading import load_dataset
from src.evaluation import cross_validate_candidates, evaluate_on_test
from src.models import get_candidate_models
from src.plotting import (plot_eda, plot_factor_effects, plot_knn_curve,
                          plot_parity, plot_polynomial_curve, plot_residuals)
from src.preprocessing import split_data

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "datasets" / "dataset_cell_growth_4factor.csv"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(DATA_PATH)
    print(f"Loaded {len(X)} samples")
    X.describe().round(3).to_csv(TABLES / "data_summary.csv")
    plot_eda(X, y, FIGURES / "eda_scatter.png")

    X_train, X_test, y_train, y_test = split_data(X, y)

    candidates = get_candidate_models()
    results = cross_validate_candidates(candidates, X_train, y_train)
    results.round(5).to_csv(TABLES / "cv_results_all_models.csv", index=False)
    plot_polynomial_curve(results, FIGURES / "polynomial_degree_curve.png")
    plot_knn_curve(results, FIGURES / "knn_k_curve.png")

    print("\nAll models by cross-validated RMSE:")
    print(results[["Model", "Train RMSE", "CV RMSE", "CV RMSE Std", "CV R2"]].round(4).to_string(index=False))

    best_name = results.loc[0, "Model"]
    best_model = candidates[best_name]["model"].fit(X_train, y_train)
    print(f"\nSelected model: {best_name}")

    test_scores = evaluate_on_test(best_model, X_test, y_test)
    pd.DataFrame([{"Model": best_name, **test_scores}]).round(5).to_csv(
        TABLES / "final_model_test_scores.csv", index=False)
    print("Test set:", {k: round(v, 4) for k, v in test_scores.items()})

    plot_parity(best_model, X_test, y_test, FIGURES / "final_parity.png")
    plot_residuals(best_model, X_test, y_test, FIGURES / "final_residuals.png")
    effects = plot_factor_effects(best_model, X, FIGURES / "factor_effects.png")
    effects.round(5).to_csv(TABLES / "factor_effects.csv", index=False)


if __name__ == "__main__":
    main()