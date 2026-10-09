"""Run the movable-wall multi-objective optimisation (paper Sec. 4-5) on both case studies."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aiobep.layout import case_study_1, case_study_2, evaluate, OFFICE  # noqa: E402
from aiobep.optimise import run_once  # noqa: E402
from aiobep.surrogate import Surrogate  # noqa: E402

# Hyper-parameters copied from paper Tables 3 and 6: (pop, p_m, eta_m, p_c)
HP = {
    "case1_simple": {"NSGA-II": (30, 0.7, 5.0, 0.1), "NSGA-III": (40, 0.7, 5.0, 0.8), "SMS-EMOA": (40, 0.7, 5.0, 0.7)},
    "case2_watson": {"NSGA-II": (50, 0.7, 1.5, 0.7), "NSGA-III": (40, 0.7, 1.5, 0.9), "SMS-EMOA": (30, 0.8, 1.5, 0.8)},
}
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=5, help="paper uses 30")
ap.add_argument("--gens", type=int, default=500)
ap.add_argument("--model", default="surrogate_ensemble", choices=["surrogate_ensemble", "surrogate_mlp"])
ap.add_argument("--algos", nargs="+", default=["NSGA-II", "NSGA-III", "SMS-EMOA"])
ap.add_argument("--jobs", type=int, default=-1)
args = ap.parse_args()

out = ROOT / "outputs" / args.model.replace("surrogate_", "opt_")
out.mkdir(parents=True, exist_ok=True)
sur = Surrogate.load(ROOT / "outputs" / "models" / f"{args.model}.joblib")


def one(sc, algo, seed):
    pop, pm, eta, pc = HP[sc.name][algo]
    r = run_once(sc, sur, algo, seed, pop=pop, n_gen=args.gens, p_m=pm, eta_m=eta, p_c=pc)
    return sc.name, algo, seed, r


def draw_layout(ax, sc, W, area, occ, T, title):
    colours = {OFFICE: "#cfe2f3"}
    for j in range(sc.n_rooms):
        c = colours.get(sc.room_types[j], "#d9ead3")
        ax.add_patch(plt.Rectangle((W[j], 0), W[j + 1] - W[j], sc.B, fc=c, ec="k"))
        ax.text((W[j] + W[j + 1]) / 2, sc.B / 2, f"R{j + 1}\n{area[j]:.1f} m²\n{occ[j]} pax\n{T[j]:.1f}°C",
                ha="center", va="center", fontsize=7)
    ax.set(xlim=(0, sc.L), ylim=(0, sc.B), title=title, yticks=[], aspect="auto")


summary = {}
for sc in (case_study_1(), case_study_2()):
    jobs = [(a, s) for a in args.algos for s in range(1, args.seeds + 1)]
    res = Parallel(n_jobs=args.jobs)(delayed(one)(sc, a, s) for a, s in jobs)
    init = evaluate(sc, sur, sc.x_init[None, 1:-1])
    f0 = init["F"][0]
    rows, fronts = [], []
    for _, algo, seed, r in res:
        F = r["F"]
        rows.append(dict(algo=algo, seed=seed, n_solutions=len(F), hypervolume_norm=r["hv"],
                         best_f_cost=F[:, 0].min() if len(F) else np.nan, best_f_tc=F[:, 1].min() if len(F) else np.nan))
        for x, f in zip(r["X"], F):
            fronts.append(dict(algo=algo, seed=seed, f_cost=f[0], f_tc=f[1], **{f"x{i+1}": v for i, v in enumerate(x)}))
    runs, front = pd.DataFrame(rows), pd.DataFrame(fronts)
    runs.to_csv(out / f"{sc.name}_runs.csv", index=False)
    front.to_csv(out / f"{sc.name}_fronts.csv", index=False)
    agg = runs.groupby("algo").agg(hv_mean=("hypervolume_norm", "mean"), hv_std=("hypervolume_norm", "std"),
                                   n_sol=("n_solutions", "mean"), f_cost=("best_f_cost", "mean"),
                                   f_tc=("best_f_tc", "mean"))
    print(f"\n== {sc.name}: initial f_cost={f0[0]:.3f} kW, f_tc={f0[1]:.3f}\n", agg.round(4).to_string())

    xcols = [c for c in front.columns if c.startswith("x")]
    A = front.loc[front.f_cost.idxmin()]            # max energy saving
    Bsol = front.loc[front.f_tc.idxmin()]           # best comfort
    sols = {"A (min energy)": A, "B (best comfort)": Bsol}
    info = dict(initial=dict(f_cost=float(f0[0]), f_tc=float(f0[1])), aggregate=agg.round(5).to_dict())
    fig, axs = plt.subplots(3, 1, figsize=(10, 7.5))
    draw_layout(axs[0], sc, init["W"][0], init["area"][0], init["occ"][0], init["T"][0],
                f"Initial: f_cost={f0[0]:.3f} kW, f_tc={f0[1]:.3f}")
    for ax, (k, s) in zip(axs[1:], sols.items()):
        r = evaluate(sc, sur, s[xcols].to_numpy(float)[None])
        fc, ft = r["F"][0]
        info[k] = dict(f_cost=float(fc), f_tc=float(ft), energy_saving_pct=float((1 - fc / f0[0]) * 100),
                       comfort_gain_pct=float((1 - ft / f0[1]) * 100), algo=s["algo"], seed=int(s["seed"]),
                       walls=[round(float(w), 2) for w in r["W"][0]], areas=r["area"][0].round(1).tolist(),
                       occupants=r["occ"][0].tolist())
        draw_layout(ax, sc, r["W"][0], r["area"][0], r["occ"][0], r["T"][0],
                    f"Solution {k}: f_cost={fc:.3f} ({info[k]['energy_saving_pct']:+.2f}% saving), "
                    f"f_tc={ft:.3f} ({info[k]['comfort_gain_pct']:+.2f}% better)")
    plt.tight_layout(); plt.savefig(out / f"{sc.name}_layouts.png", dpi=130); plt.close()
    print({k: {kk: vv for kk, vv in v.items() if kk in ("energy_saving_pct", "comfort_gain_pct")} for k, v in info.items() if k[0] in "AB"})
    json.dump(info, open(out / f"{sc.name}_summary.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for algo, g in front.groupby("algo"):
        ax.scatter(g.f_cost, g.f_tc, s=10, alpha=.5, label=algo)
    ax.scatter(*f0, c="k", marker="*", s=150, label="initial layout")
    ax.set(xlabel="f_cost  [kW, surrogate]", ylabel="f_tc  [mean |PMV| per occupant]", title=f"{sc.name}: non-dominated fronts ({args.seeds} seeds)")
    ax.legend(); plt.tight_layout(); plt.savefig(out / f"{sc.name}_pareto.png", dpi=130); plt.close()
