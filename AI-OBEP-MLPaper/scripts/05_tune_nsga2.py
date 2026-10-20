"""Step 5 - tune NSGA-II hyperparameters with PSO, GA and HHO (Lyu et al. used Optuna).

Search space (paper Table A.8 widened, + setpoint mutation sigma):
  pop_size 10-100, mutation prob 0.05-1, wall mutation sd eta_m 0.2-6 m,
  crossover prob 0.1-1, setpoint mutation sd sigma_t 0.05-1.5 degC.
Fitness = -mean normalised hypervolume over 3 tuning seeds (101-103) at a FIXED
budget of function evaluations, so a bigger population cannot win by simply
spending more evaluations. Tuning seeds are disjoint from the final seeds 1-30.
Usage: python scripts/05_tune_nsga2.py [--budget 50] [--mh-seeds 2] [--nsga-evals 3000]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from obep_mlpaper.layout import case_study_1, case_study_2  # noqa: E402
from obep_mlpaper.metaheuristics import ALGORITHMS, Param, SearchSpace  # noqa: E402
from obep_mlpaper.nsga import run_nsga2  # noqa: E402
from obep_mlpaper.surrogate import Surrogate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--budget", type=int, default=50)
ap.add_argument("--mh-seeds", type=int, default=2)
ap.add_argument("--nsga-evals", type=int, default=3000)
ap.add_argument("--pop", type=int, default=10)
args = ap.parse_args()

SPACE = SearchSpace([
    Param("pop_size", 10, 100, "int"),
    Param("p_m", 0.05, 1.0),
    Param("eta_m", 0.2, 6.0, "log"),
    Param("p_c", 0.1, 1.0),
    Param("sigma_t", 0.05, 1.5, "log"),
])
TUNE_SEEDS = (101, 102, 103)
sur = Surrogate.load(ROOT / "outputs" / "models" / "surrogate.joblib").single_thread()
out = ROOT / "outputs" / "nsga2"
out.mkdir(parents=True, exist_ok=True)

result = {}
runs, traces = [], {}
for sc in (case_study_1(), case_study_2()):
    def fitness(hp, sc=sc):
        hvs = Parallel(n_jobs=2)(delayed(run_nsga2)(sc, sur, hp, s, args.nsga_evals) for s in TUNE_SEEDS)
        return -float(np.mean([r["hv"] for r in hvs]))

    for algo, fn in ALGORITHMS.items():
        for seed in range(args.mh_seeds):
            t0 = time.time()
            kw = {"PSO": "n_particles", "GA": "pop_size", "HHO": "n_hawks"}[algo]
            r = fn(fitness, SPACE, budget=args.budget, seed=seed, **{kw: args.pop})
            runs.append(dict(case=sc.name, algo=algo, seed=seed, best_hv=-r["best_f"],
                             minutes=(time.time() - t0) / 60, params=json.dumps(r["best_params"])))
            traces[f"{sc.name}_{algo}_s{seed}"] = -r["trace"]
            print(f"{sc.name} {algo} seed={seed}: HV={-r['best_f']:.4f} {r['best_params']} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)

runs = pd.DataFrame(runs)
runs.to_csv(out / "nsga2_tuning_runs.csv", index=False)
L = max(len(t) for t in traces.values())
pd.DataFrame({k: np.pad(t, (0, L - len(t)), mode="edge") for k, t in traces.items()}).to_csv(
    out / "nsga2_tuning_convergence.csv", index_label="evaluation")
for case, grp in runs.groupby("case"):
    result[case] = {a: json.loads(grp[grp.algo == a].sort_values("best_hv").iloc[-1].params) for a in ALGORITHMS}
(out / "nsga2_tuned_hyperparameters.json").write_text(json.dumps(result, indent=2))
print(runs.groupby(["case", "algo"]).best_hv.agg(["mean", "std", "max"]).round(4))
