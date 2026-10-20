"""NSGA-II for the movable-wall + setpoint problem (pymoo), plus quality indicators."""
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.operators.crossover.sbx import SBX
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from .layout import Scenario, evaluate

HV_REF = 1.5   # reference point (1.5, 1.5) on objectives normalised by the 24 degC baseline

# Paper hyperparameters for NSGA-II (Tables 3 and 6) + setpoint mutation sigma (new).
PAPER_HP = {
    "case1_simple": dict(pop_size=30, p_m=0.7, eta_m=5.0, p_c=0.1, sigma_t=0.5),
    "case2_watson": dict(pop_size=50, p_m=0.7, eta_m=1.5, p_c=0.7, sigma_t=0.5),
}


class LayoutProblem(Problem):
    def __init__(self, sc: Scenario, surrogate):
        self.sc, self.surrogate = sc, surrogate
        super().__init__(n_var=sc.n_walls + sc.n_zones, n_obj=2, n_ieq_constr=sc.n_rooms + 1,
                         xl=sc.xl, xu=sc.xu)

    def _evaluate(self, X, out, *args, **kwargs):
        r = evaluate(self.sc, self.surrogate, X)
        out["F"], out["G"] = r["F"], r["G"]


class BaselineSampling(Sampling):
    """Paper: every initial solution equals the existing configuration (here at 24 degC)."""

    def __init__(self, x0):
        super().__init__()
        self.x0 = np.asarray(x0, float)

    def _do(self, problem, n_samples, **kwargs):
        return np.tile(self.x0, (n_samples, 1))


class WallRepair(Repair):
    """Clip into bounds and re-order walls (wall re-ordering of Algorithm 6)."""

    def __init__(self, n_walls):
        super().__init__()
        self.n_walls = n_walls

    def _do(self, problem, X, **kwargs):
        X = np.clip(X, problem.xl, problem.xu)
        X[:, :self.n_walls] = np.sort(X[:, :self.n_walls], axis=1)
        return X


class WallSetpointMutation(Mutation):
    """Algorithm 6 on the wall genes (Gaussian, sd eta_m metres, prob p_m, clip, sort)
    plus Gaussian mutation of the zone setpoints (sd sigma_t degC, prob p_m)."""

    def __init__(self, sc: Scenario, p_m, eta_m, sigma_t):
        super().__init__(prob=1.0)
        self.sc, self.p_m, self.eta_m, self.sigma_t = sc, p_m, eta_m, sigma_t

    def _do(self, problem, X, **kwargs):
        X = X.copy()
        nw = self.sc.n_walls
        sd = np.r_[np.full(nw, self.eta_m), np.full(self.sc.n_zones, self.sigma_t)]
        hit = np.random.random(X.shape) < self.p_m
        X = X + hit * np.random.normal(0.0, 1.0, X.shape) * sd
        X = np.clip(X, problem.xl, problem.xu)
        X[:, :nw] = np.sort(X[:, :nw], axis=1)
        return X


class HVTrace(Callback):
    """Normalised hypervolume of the feasible non-dominated set every ``every`` generations."""

    def __init__(self, f0, every=10):
        super().__init__()
        self.f0, self.every, self.rows = f0, every, []
        self.hv = HV(ref_point=np.full(2, HV_REF))

    def notify(self, algorithm):
        if algorithm.n_gen % self.every and algorithm.n_gen != 1:
            return
        opt = algorithm.opt
        F = opt.get("F")
        feas = opt.get("feasible").ravel()
        hv = float(self.hv(F[feas] / self.f0)) if feas.any() else 0.0
        self.rows.append((algorithm.n_gen, algorithm.evaluator.n_eval, hv))


def make_nsga2(sc, pop_size, p_m, eta_m, p_c, sigma_t):
    return NSGA2(pop_size=int(pop_size), sampling=BaselineSampling(sc.baseline()),
                 crossover=SBX(prob=p_c, eta=15), mutation=WallSetpointMutation(sc, p_m, eta_m, sigma_t),
                 repair=WallRepair(sc.n_walls), eliminate_duplicates=True)


def nondominated(F):
    if len(F) == 0:
        return np.array([], int)
    return NonDominatedSorting().do(F, only_non_dominated_front=True)


def run_nsga2(sc, surrogate, hp: dict, seed: int, n_evals: int = 25_000, trace_every=None):
    """One NSGA-II run with a fixed budget of function evaluations.
    Returns the feasible non-dominated front (raw objectives) and its decision vectors."""
    problem = LayoutProblem(sc, surrogate)
    f0 = evaluate(sc, surrogate, sc.baseline()[None])["F"][0]
    cb = HVTrace(f0, trace_every) if trace_every else None
    kw = {"callback": cb} if cb else {}
    res = minimize(problem, make_nsga2(sc, **hp), ("n_eval", n_evals), seed=seed, verbose=False, **kw)
    pop = res.pop
    F, X, G = pop.get("F"), pop.get("X"), pop.get("G")
    feas = (G <= 1e-9).all(axis=1)
    F, X = F[feas], X[feas]
    nd = nondominated(F)
    F, X = F[nd], X[nd]
    # remove exact duplicates in objective space
    _, keep = np.unique(np.round(F, 9), axis=0, return_index=True)
    F, X = F[np.sort(keep)], X[np.sort(keep)]
    hv = float(HV(ref_point=np.full(2, HV_REF))(F / f0)) if len(F) else 0.0
    return dict(F=F, X=X, f0=f0, hv=hv, trace=(cb.rows if cb else None))


def igd_plus(F, ref_front, f0):
    """IGD+ (Ishibuchi et al., 2015) on normalised objectives; 0 = reaches the reference front."""
    return float(IGDPlus(ref_front / f0)(F / f0))
