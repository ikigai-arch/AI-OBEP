"""Surrogate models for room-level HVAC energy (sklearn stand-in for AutoGluon).

The ML paper uses AutoGluon-Tabular, whose best model is a weighted ensemble of
tree models and neural nets. Here we train a small model zoo (including the MLP
suggested for AI-OBEP) and build the same kind of weighted ensemble.
"""
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import nnls
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def model_zoo(seed: int = 0) -> dict:
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-3, early_stopping=True,
                       n_iter_no_change=20, max_iter=600, random_state=seed)
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge()),
        "RandomForest": RandomForestRegressor(150, min_samples_leaf=5, n_jobs=-1, random_state=seed),
        "ExtraTrees": ExtraTreesRegressor(200, min_samples_leaf=5, n_jobs=-1, random_state=seed),
        "HistGB": HistGradientBoostingRegressor(random_state=seed),
        "MLP": TransformedTargetRegressor(make_pipeline(StandardScaler(), mlp), transformer=StandardScaler()),
    }


def metrics(y, p, mape_floor: float = 0.5) -> dict:
    """R2, RMSE, MAE (Wh/m2/5min) and MAPE.

    MAPE explodes on near-zero targets (HVAC off), so it is computed only where
    y >= ``mape_floor`` Wh/m2; treat it as indicative, R2/RMSE/MAE are the core.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = y >= mape_floor
    return dict(
        R2=float(r2_score(y, p)),
        RMSE=float(np.sqrt(np.mean((y - p) ** 2))),
        MAE=float(mean_absolute_error(y, p)),
        MAPE_pct=float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100) if m.any() else float("nan"),
    )


class WeightedEnsemble:
    """Non-negative least-squares blend of fitted models (fit on validation)."""

    def __init__(self, models: dict):
        self.models, self.weights = models, None

    def fit_weights(self, X, y):
        P = np.column_stack([m.predict(X) for m in self.models.values()])
        w, _ = nnls(P, np.asarray(y, float))
        self.weights = w / w.sum()
        return {k: round(float(v), 3) for k, v in zip(self.models, self.weights)}

    def predict(self, X):
        P = np.column_stack([m.predict(X) for m in self.models.values()])
        return np.clip(P @ self.weights, 0, None)


class Surrogate:
    """Thin wrapper: ndarray (n, 7) in ``FEATURES`` order -> Wh/m2 per 5 min."""

    def __init__(self, model, features, meta=None):
        self.model, self.features, self.meta = model, list(features), meta or {}

    def predict(self, X):
        return np.clip(self.model.predict(np.asarray(X, float)), 0, None)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "Surrogate":
        return joblib.load(path)
