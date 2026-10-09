"""Multi-objective optimisation of the movable-wall problem with pymoo."""
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.sbx import SBX
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from .layout import Scenario, evaluate


class WallProblem(Problem):
    """x = inner wall positions; F = [f_cost, f_tc]; G = constraint violations (<=0 feasible)."""

    def __init__(self, sc: Scenario, surrogate):
        self.sc, self.surrogate = sc, surrogate
        n = sc.n_rooms - 1
        super().__init__(n_var=n, n_obj=2, n_ieq_constr=sc.n_rooms + 1, xl=0.0, xu=sc.L)

    def _evaluate(self, X, out, *args, **kwargs):
        r = evaluate(self.sc, self.surrogate, X)
        out["F"], out["G"] = r["F"], r["G"]


class InitialSampling(Sampling):
    """Paper: every initial solution equals the existing layout."""

    def __init__(self, x0):
        super().__init__()
        self.x0 = np.asarray(x0, float)

    def _do(self, problem, n_samples, **kwargs):
        return np.tile(self.x0, (n_samples, 1))


class SortRepair(Repair):
    """Clip into [0, L] and re-order walls so adjacent walls never cross."""

    def _do(self, problem, X, **kwargs):
        return np.sort(np.clip(X, problem.xl, problem.xu), axis=1)


class CustomMutation(Mutation):
    """Algorithm 6: Gaussian noise on movable walls, clip, then sort (wall re-ordering).
    Immovable walls are reset to their initial position."""

    def __init__(self, sc: Scenario, p_m: float, eta_m: float):
        super().__init__(prob=1.0)
        self.sc, self.p_m, self.eta_m = sc, p_m, eta_m
        self.mov = np.asarray(sc.movable)
        self.x0 = sc.x_init[1:-1]

    def _do(self, problem, X, **kwargs):
        X = X.copy()
        hit = (np.random.random(X.shape) < self.p_m) & self.mov
        X = X + hit * np.random.normal(0.0, self.eta_m, X.shape)
        X = np.clip(X, 0.0, self.sc.L)
        X[:, ~self.mov] = self.x0[~self.mov]
        return np.sort(X, axis=1)


def make_algorithm(name: str, sc: Scenario, pop: int, p_m: float, eta_m: float, p_c: float):
    common = dict(sampling=InitialSampling(sc.x_init[1:-1]),
                  mutation=CustomMutation(sc, p_m, eta_m), repair=SortRepair(),
                  eliminate_duplicates=True)
    cx = SBX(prob=p_c, eta=15)
    if name == "NSGA-II":
        return NSGA2(pop_size=pop, crossover=cx, **common)
    if name == "NSGA-III":
        ref = get_reference_directions("das-dennis", 2, n_partitions=12)
        return NSGA3(ref_dirs=ref, pop_size=pop, crossover=cx, **common)
    if name == "SMS-EMOA":
        return SMSEMOA(pop_size=pop, crossover=cx, **common)
    raise ValueError(name)


def nondominated_feasible(F, G):
    """Feasible, non-dominated subset indices of a result set."""
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    feas = np.where((G <= 1e-9).all(axis=1))[0]
    if len(feas) == 0:
        return feas
    return feas[NonDominatedSorting().do(F[feas], only_non_dominated_front=True)]


def run_once(sc, surrogate, algo, seed, pop=50, n_gen=500, p_m=0.7, eta_m=1.5, p_c=0.7,
             ref_mult=1.2):
    """Run one optimiser; return front, layouts and normalised hypervolume.

    HV is computed on objectives divided by the initial layout's values, with
    reference point (ref_mult, ref_mult) - unitless, comparable across algorithms.
    """
    problem = WallProblem(sc, surrogate)
    f0 = evaluate(sc, surrogate, sc.x_init[None, 1:-1])["F"][0]
    res = minimize(problem, make_algorithm(algo, sc, pop, p_m, eta_m, p_c),
                   ("n_gen", n_gen), seed=seed, verbose=False, save_history=False)
    if res.X is None:
        return dict(F=np.empty((0, 2)), X=np.empty((0, problem.n_var)), hv=0.0, f0=f0)
    X, F = np.atleast_2d(res.X), np.atleast_2d(res.F)
    hv = float(HV(ref_point=np.full(2, ref_mult))(F / f0))
    return dict(F=F, X=X, hv=hv, f0=f0)
