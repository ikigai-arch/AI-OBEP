"""Room-level HVAC energy surrogate: model zoo, metrics, metaheuristic fitness."""
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from lightgbm import LGBMRegressor
from scipy.optimize import nnls
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metaheuristics import Param, SearchSpace

N_JOBS = 2


def model_zoo(seed: int = 0) -> dict:
    """Default-hyperparameter models (stand-in for the AutoGluon-Tabular zoo)."""
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-3, early_stopping=True,
                       n_iter_no_change=20, max_iter=1000, random_state=seed)
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge()),
        "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(10, weights="distance")),
        "RandomForest": RandomForestRegressor(300, min_samples_leaf=2, n_jobs=N_JOBS, random_state=seed),
        "ExtraTrees": ExtraTreesRegressor(400, min_samples_leaf=2, n_jobs=N_JOBS, random_state=seed),
        "LightGBM": LGBMRegressor(n_estimators=300, learning_rate=0.05, verbose=-1, random_state=seed,
                                  n_jobs=N_JOBS),
        "XGBoost": xgb.XGBRegressor(n_jobs=N_JOBS, random_state=seed, tree_method="hist"),
        "MLP": TransformedTargetRegressor(make_pipeline(StandardScaler(), mlp), transformer=StandardScaler()),
    }


# XGBoost search space tuned by PSO / GA / HHO.
XGB_SPACE = SearchSpace([
    Param("n_estimators", 100, 800, "int"),
    Param("learning_rate", 0.01, 0.3, "log"),
    Param("max_depth", 2, 8, "int"),
    Param("min_child_weight", 1, 20, "log"),
    Param("subsample", 0.5, 1.0),
    Param("colsample_bytree", 0.4, 1.0),
    Param("gamma", 1e-3, 5.0, "log"),
    Param("reg_lambda", 1e-3, 20.0, "log"),
    Param("reg_alpha", 1e-3, 5.0, "log"),
])


def make_xgb(params: dict, seed: int = 0):
    return xgb.XGBRegressor(**params, n_jobs=N_JOBS, random_state=seed, tree_method="hist")


def metrics(y, p, mape_floor: float = None) -> dict:
    """R2, RMSE, MAE, MAPE plus ASHRAE Guideline 14 CV(RMSE) and NMBE (%).

    MAPE explodes on near-zero targets (HVAC off at night), so it is computed only
    where y >= ``mape_floor`` (default: 10 % of the target maximum)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    floor = 0.1 * y.max() if mape_floor is None else mape_floor
    m = y >= floor
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    return dict(
        R2=float(r2_score(y, p)), RMSE=rmse, MAE=float(mean_absolute_error(y, p)),
        MAPE_pct=float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100),
        CVRMSE_pct=100 * rmse / float(y.mean()),
        NMBE_pct=100 * float(np.sum(y - p)) / (len(y) * float(y.mean())),
    )


def grouped_cv_predict(make_model, X, y, groups, n_splits=5):
    """Out-of-fold predictions with whole days held out in each fold."""
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = make_model()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof


def xgb_cv_fitness(X, y, groups, n_splits=5, seed=0):
    """Metaheuristic objective: grouped-CV RMSE of XGBoost (lower is better)."""
    def f(params):
        oof = grouped_cv_predict(lambda: make_xgb(params, seed), X, y, groups, n_splits)
        return float(np.sqrt(np.mean((y - oof) ** 2)))
    return f


class WeightedEnsemble:
    """Non-negative least-squares blend (AutoGluon 'WeightedEnsemble_L2' analogue).
    Weights are fit on out-of-fold predictions; members are then refit on all data."""

    def __init__(self, makers: dict):
        self.makers, self.models, self.weights = makers, {}, None

    def fit(self, X, y, groups):
        oof = np.column_stack([grouped_cv_predict(mk, X, y, groups) for mk in self.makers.values()])
        w, _ = nnls(oof, y)
        self.weights = w / w.sum()
        self.models = {k: mk().fit(X, y) for k, mk in self.makers.items()}
        return self

    def weight_dict(self):
        return {k: round(float(w), 3) for k, w in zip(self.makers, self.weights)}

    def predict(self, X):
        P = np.column_stack([m.predict(X) for m in self.models.values()])
        return P @ self.weights


class Surrogate:
    """ndarray (n, len(features)) in ``features`` order -> HVAC power density [W/m2] (>= 0)."""

    def __init__(self, model, features, meta=None):
        self.model, self.features, self.meta = model, list(features), meta or {}

    def predict(self, X):
        return np.clip(self.model.predict(np.asarray(X, float)), 0, None)

    def single_thread(self):
        """Use 1 thread per process (avoids OpenMP oversubscription in parallel NSGA-II runs)."""
        if hasattr(self.model, "set_params") and "n_jobs" in self.model.get_params():
            self.model.set_params(n_jobs=1)
        return self

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "Surrogate":
        return joblib.load(path)
