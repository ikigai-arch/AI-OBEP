"""Step 2 - tune the XGBoost surrogate with PSO, GA and HHO (equal budget).

Fitness = 5-fold grouped (leave-days-out) CV RMSE on the training days.
Held-out test days are never seen during tuning. Outputs -> outputs/surrogate/.
Usage: python scripts/02_tune_surrogate.py [--budget 200] [--seeds 3]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from obep_mlpaper.data import FEATURES, TARGET, day_split, hourly_dataset  # noqa: E402
from obep_mlpaper.metaheuristics import ALGORITHMS  # noqa: E402
from obep_mlpaper.surrogate import XGB_SPACE, make_xgb, metrics, xgb_cv_fitness  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--budget", type=int, default=200)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--pop", type=int, default=15)
args = ap.parse_args()

out = ROOT / "outputs" / "surrogate"
out.mkdir(parents=True, exist_ok=True)

df = hourly_dataset()
train, test = day_split(df)
X, y, g = df[FEATURES].to_numpy(), df[TARGET].to_numpy(), df["day"].to_numpy()
Xtr, ytr, gtr = X[train], y[train], g[train]
fitness = xgb_cv_fitness(Xtr, ytr, gtr)

runs, traces, logs = [], {}, []
for algo, fn in ALGORITHMS.items():
    for seed in range(args.seeds):
        t0 = time.time()
        kw = {"PSO": "n_particles", "GA": "pop_size", "HHO": "n_hawks"}[algo]
        r = fn(fitness, XGB_SPACE, budget=args.budget, seed=seed, **{kw: args.pop})
        m = make_xgb(r["best_params"]).fit(Xtr, ytr)
        te = metrics(y[test], np.clip(m.predict(X[test]), 0, None))
        runs.append(dict(algo=algo, seed=seed, cv_rmse=r["best_f"], n_evals=r["n_evals"],
                         minutes=(time.time() - t0) / 60, **{f"test_{k}": v for k, v in te.items()},
                         params=json.dumps(r["best_params"])))
        traces[f"{algo}_s{seed}"] = r["trace"]
        logs += [dict(algo=algo, seed=seed, **row) for row in r["log"]]
        print(f"{algo} seed={seed}: CV-RMSE={r['best_f']:.4f}  test R2={te['R2']:.4f} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)

runs = pd.DataFrame(runs)
runs.to_csv(out / "metaheuristic_runs.csv", index=False)
pd.DataFrame(logs).to_csv(out / "metaheuristic_eval_log.csv", index=False)
L = max(len(t) for t in traces.values())
pd.DataFrame({k: np.pad(t, (0, L - len(t)), mode="edge") for k, t in traces.items()}).to_csv(
    out / "metaheuristic_convergence.csv", index_label="evaluation")

summary = runs.groupby("algo").agg(cv_rmse_mean=("cv_rmse", "mean"), cv_rmse_std=("cv_rmse", "std"),
                                   cv_rmse_best=("cv_rmse", "min"), test_R2_mean=("test_R2", "mean"),
                                   test_RMSE_mean=("test_RMSE", "mean"), minutes=("minutes", "mean"))
summary.to_csv(out / "metaheuristic_summary.csv")
print(summary.round(4))
best = runs.loc[runs.cv_rmse.idxmin()]
(out / "best_xgb_params.json").write_text(json.dumps(dict(
    algo=best.algo, seed=int(best.seed), cv_rmse=float(best.cv_rmse),
    params=json.loads(best.params), per_algo={a: json.loads(runs[runs.algo == a].sort_values("cv_rmse").iloc[0].params)
                                              for a in ALGORITHMS}), indent=2))
print("best:", best.algo, best.seed, best.params)
