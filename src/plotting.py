from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_eda(X: pd.DataFrame, y: pd.Series, path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.ravel(), X.columns):
        ax.scatter(X[col], y, s=8, alpha=0.5)
        ax.set_xlabel(col)
        ax.set_ylabel(y.name)
    _save(fig, path)


def plot_polynomial_curve(results: pd.DataFrame, path: Path):
    df = results[results["Family"] == "Polynomial"].sort_values("Degree")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["Degree"], df["Train RMSE"], marker="o", label="Train RMSE")
    ax.errorbar(df["Degree"], df["CV RMSE"], yerr=df["CV RMSE Std"],
                marker="o", capsize=4, label="CV RMSE")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("RMSE [d^-1]")
    ax.set_title("Polynomial regression: error vs degree")
    ax.legend()
    _save(fig, path)


def plot_knn_curve(results: pd.DataFrame, path: Path):
    df = results[results["Family"] == "kNN"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for w, group in df.groupby("Weights"):
        group = group.sort_values("k")
        ax.errorbar(group["k"], group["CV RMSE"], yerr=group["CV RMSE Std"],
                    marker="o", capsize=4, label=f"CV RMSE ({w})")
    ax.set_xscale("log")
    ax.set_xlabel("Number of neighbours k")
    ax.set_ylabel("RMSE [d^-1]")
    ax.set_title("k-NN regression: error vs k")
    ax.legend()
    _save(fig, path)


def plot_parity(model, X_test, y_test, path: Path):
    y_pred = model.predict(X_test)
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, s=10, alpha=0.6)
    ax.plot([lo, hi], [lo, hi], "k--", label="Perfect prediction")
    ax.set_xlabel("Actual growth rate [d^-1]")
    ax.set_ylabel("Predicted growth rate [d^-1]")
    ax.set_title("Final model: predicted vs actual (test set)")
    ax.legend()
    _save(fig, path)


def plot_residuals(model, X_test, y_test, path: Path):
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, y_test - y_pred, s=10, alpha=0.6)
    ax.axhline(0, color="k", linestyle="--")
    ax.set_xlabel("Predicted growth rate [d^-1]")
    ax.set_ylabel("Residual [d^-1]")
    ax.set_title("Final model: residuals (test set)")
    _save(fig, path)


def plot_factor_effects(model, X: pd.DataFrame, path: Path, n_points: int = 100) -> pd.DataFrame:
    medians = X.median()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    curves = []
    for ax, col in zip(axes.ravel(), X.columns):
        grid = np.linspace(X[col].min(), X[col].max(), n_points)
        X_sweep = pd.DataFrame([medians] * n_points).reset_index(drop=True)
        X_sweep[col] = grid
        pred = model.predict(X_sweep)
        ax.plot(grid, pred)
        ax.set_xlabel(col)
        ax.set_ylabel("Predicted growth rate [d^-1]")
        curves.append(pd.DataFrame({"Factor": col, "Value": grid, "Predicted Growth Rate [d^-1]": pred}))
    fig.suptitle("Effect of each factor (others held at median)")
    _save(fig, path)
    return pd.concat(curves, ignore_index=True)