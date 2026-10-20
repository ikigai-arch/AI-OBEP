"""Step 2b - PSO / GA / HHO tuning under the exact Lyu et al. (2025) protocol.

office-1, first 12 weekdays, 5-min, outlier smoothing, 7 paper features,
random 70/15/15 row split (AutoGluon-Tabular default). Fitness = validation RMSE;
the test rows are only used for the final report, comparable to the paper's Table 1.
Outputs -> outputs/surrogate/paper_protocol_tuned.csv
Usage: python scripts/02b_tune_paper_protocol.py [--budget 100] [--seeds 2]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from obep_mlpaper.data import FEATURES_PAPER, five_minute_dataset  # noqa: E402
from obep_mlpaper.metaheuristics import ALGORITHMS  # noqa: E402
from obep_mlpaper.surrogate import XGB_SPACE, make_xgb, metrics, model_zoo  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--budget", type=int, default=100)
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--pop", type=int, default=10)
args = ap.parse_args()
out = ROOT / "outputs" / "surrogate"

d = five_minute_dataset(n_days=12, smooth=True)
idx = np.random.RandomState(0).permutation(len(d))
n = len(idx)
itr, iva, ite = idx[:int(.7 * n)], idx[int(.7 * n):int(.85 * n)], idx[int(.85 * n):]
X, y = d[FEATURES_PAPER].to_numpy(), d["target_kwh"].to_numpy()


def fitness(params):
    m = make_xgb(params).fit(X[itr], y[itr])
    return float(np.sqrt(np.mean((y[iva] - m.predict(X[iva])) ** 2)))


rows = []
m = model_zoo()["XGBoost"].fit(X[itr], y[itr])
rows.append(dict(model="XGBoost (default)", val_RMSE=float(np.sqrt(np.mean((y[iva] - m.predict(X[iva])) ** 2))),
                 **metrics(y[ite], np.clip(m.predict(X[ite]), 0, None))))
for algo, fn in ALGORITHMS.items():
    for seed in range(args.seeds):
        kw = {"PSO": "n_particles", "GA": "pop_size", "HHO": "n_hawks"}[algo]
        r = fn(fitness, XGB_SPACE, budget=args.budget, seed=seed, **{kw: args.pop})
        m = make_xgb(r["best_params"]).fit(X[itr], y[itr])
        rows.append(dict(model=f"XGBoost-{algo}", seed=seed, val_RMSE=r["best_f"],
                         **metrics(y[ite], np.clip(m.predict(X[ite]), 0, None)), params=json.dumps(r["best_params"])))
        print(rows[-1]["model"], seed, f"val RMSE={r['best_f']:.4f} test R2={rows[-1]['R2']:.4f}", flush=True)
rows.append(dict(model="Lyu et al. 2025 WeightedEnsemble_L2", R2=0.933, MAPE_pct=26.094))
rows.append(dict(model="Lyu et al. 2025 LSTM", R2=0.921, MAPE_pct=19.358))
res = pd.DataFrame(rows)
res.to_csv(out / "paper_protocol_tuned.csv", index=False)
print(res.groupby("model", sort=False)[["val_RMSE", "R2", "MAPE_pct", "CVRMSE_pct"]].mean().round(4).to_string())
