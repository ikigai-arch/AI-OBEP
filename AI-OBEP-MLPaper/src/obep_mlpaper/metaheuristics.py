"""Three metaheuristics for box-constrained minimisation: PSO, GA and HHO.

All work in the unit hypercube [0, 1]^d (a ``SearchSpace`` decodes to real
hyperparameters) and share an evaluation budget, so the comparison is fair:
the run stops as soon as ``budget`` objective calls have been made.

References
- PSO: Kennedy & Eberhart (1995); inertia-weight form of Shi & Eberhart (1998).
- GA : real-coded GA with tournament selection, SBX crossover (Deb & Agrawal,
       1995), Gaussian mutation and elitism.
- HHO: Harris Hawks Optimisation, Heidari et al. (2019), Future Gener. Comput. Syst. 97, 849-872.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Param:
    name: str
    low: float
    high: float
    kind: str = "float"   # float | int | log


class SearchSpace:
    def __init__(self, params):
        self.params = list(params)
        self.dim = len(self.params)

    def decode(self, u):
        u = np.clip(np.asarray(u, float), 0.0, 1.0)
        out = {}
        for p, v in zip(self.params, u):
            if p.kind == "log":
                val = math.exp(math.log(p.low) + v * (math.log(p.high) - math.log(p.low)))
            else:
                val = p.low + v * (p.high - p.low)
            out[p.name] = int(round(val)) if p.kind == "int" else float(val)
        return out


class Budget:
    """Wraps the objective: counts calls, caches decoded duplicates, records the
    best-so-far trace (one entry per evaluation) and stops at ``budget``."""

    class Exhausted(Exception):
        pass

    def __init__(self, func, space: SearchSpace, budget: int):
        self.func, self.space, self.budget = func, space, budget
        self.n, self.best_f, self.best_u = 0, np.inf, None
        self.trace, self.cache, self.log = [], {}, []

    def __call__(self, u):
        params = self.space.decode(u)
        key = tuple(sorted((k, round(v, 6)) for k, v in params.items()))
        if key in self.cache:
            return self.cache[key]
        if self.n >= self.budget:
            raise Budget.Exhausted
        f = float(self.func(params))
        self.n += 1
        self.cache[key] = f
        self.log.append(dict(eval=self.n, f=f, **params))
        if f < self.best_f:
            self.best_f, self.best_u = f, np.clip(np.asarray(u, float), 0, 1).copy()
        self.trace.append(self.best_f)
        return f


def _result(b: Budget, name):
    return dict(algo=name, best_f=b.best_f, best_params=b.space.decode(b.best_u),
                trace=np.array(b.trace), log=b.log, n_evals=b.n)


# ---------------------------------------------------------------- PSO
def pso(func, space, budget=200, n_particles=15, w_max=0.9, w_min=0.4, c1=1.5, c2=1.5,
        v_max=0.2, seed=0):
    """Global-best PSO with linearly decreasing inertia weight."""
    rng = np.random.default_rng(seed)
    b = Budget(func, space, budget)
    d = space.dim
    X = rng.random((n_particles, d))
    V = rng.uniform(-v_max, v_max, (n_particles, d))
    n_iter = max(1, budget // n_particles)
    try:
        P, Pf = X.copy(), np.array([b(x) for x in X])
        g = P[Pf.argmin()].copy()
        for it in range(n_iter):
            w = w_max - (w_max - w_min) * it / max(1, n_iter - 1)
            r1, r2 = rng.random((n_particles, d)), rng.random((n_particles, d))
            V = np.clip(w * V + c1 * r1 * (P - X) + c2 * r2 * (g - X), -v_max, v_max)
            X = np.clip(X + V, 0, 1)
            for i in range(n_particles):
                f = b(X[i])
                if f < Pf[i]:
                    P[i], Pf[i] = X[i].copy(), f
            g = P[Pf.argmin()].copy()
    except Budget.Exhausted:
        pass
    return _result(b, "PSO")


# ---------------------------------------------------------------- GA
def ga(func, space, budget=200, pop_size=15, p_c=0.9, eta_c=15.0, p_m=None, sigma_m=0.1,
       n_elite=2, tournament=3, seed=0):
    """Real-coded GA: tournament selection, SBX crossover, Gaussian mutation, elitism."""
    rng = np.random.default_rng(seed)
    b = Budget(func, space, budget)
    d = space.dim
    p_m = 1.0 / d if p_m is None else p_m

    def select(F):
        idx = rng.integers(0, len(F), tournament)
        return idx[np.argmin(F[idx])]

    def sbx(a, c):
        if rng.random() > p_c:
            return a.copy(), c.copy()
        u = rng.random(d)
        beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)), (1 / (2 * (1 - u))) ** (1 / (eta_c + 1)))
        c1 = 0.5 * ((1 + beta) * a + (1 - beta) * c)
        c2 = 0.5 * ((1 - beta) * a + (1 + beta) * c)
        return np.clip(c1, 0, 1), np.clip(c2, 0, 1)

    def mutate(x):
        m = rng.random(d) < p_m
        return np.clip(x + m * rng.normal(0, sigma_m, d), 0, 1)

    try:
        X = rng.random((pop_size, d))
        F = np.array([b(x) for x in X])
        while True:
            order = np.argsort(F)
            children = [X[i].copy() for i in order[:n_elite]]
            child_f = [F[i] for i in order[:n_elite]]
            while len(children) < pop_size:
                c1, c2 = sbx(X[select(F)], X[select(F)])
                for c in (mutate(c1), mutate(c2)):
                    if len(children) < pop_size:
                        children.append(c)
                        child_f.append(b(c))
            X, F = np.array(children), np.array(child_f)
    except Budget.Exhausted:
        pass
    return _result(b, "GA")


# ---------------------------------------------------------------- HHO
def _levy(rng, d, beta=1.5):
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
             / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    return 0.01 * rng.normal(0, sigma, d) / np.abs(rng.normal(0, 1, d)) ** (1 / beta)


def hho(func, space, budget=200, n_hawks=15, seed=0):
    """Harris Hawks Optimisation (Heidari et al., 2019), standard update rules:
    exploration (|E|>=1), soft / hard besiege, and the two rapid-dive variants."""
    rng = np.random.default_rng(seed)
    b = Budget(func, space, budget)
    d = space.dim
    # each iteration costs between n_hawks and ~2*n_hawks evaluations
    T = max(1, int(budget / (1.4 * n_hawks)))
    try:
        X = rng.random((n_hawks, d))
        F = np.array([b(x) for x in X])
        rabbit, rabbit_f = X[F.argmin()].copy(), F.min()
        for t in range(T):
            E1 = 2 * (1 - t / T)
            for i in range(n_hawks):
                E = E1 * (2 * rng.random() - 1)
                J = 2 * (1 - rng.random())
                if abs(E) >= 1:                                   # exploration
                    q = rng.random()
                    if q >= 0.5:
                        Xr = X[rng.integers(n_hawks)]
                        new = Xr - rng.random() * np.abs(Xr - 2 * rng.random() * X[i])
                    else:
                        new = (rabbit - X.mean(0)) - rng.random() * rng.random(d)
                    X[i] = np.clip(new, 0, 1)
                    F[i] = b(X[i])
                else:                                              # exploitation
                    r = rng.random()
                    if r >= 0.5 and abs(E) >= 0.5:                 # soft besiege
                        X[i] = np.clip((rabbit - X[i]) - E * np.abs(J * rabbit - X[i]), 0, 1)
                        F[i] = b(X[i])
                    elif r >= 0.5:                                 # hard besiege
                        X[i] = np.clip(rabbit - E * np.abs(rabbit - X[i]), 0, 1)
                        F[i] = b(X[i])
                    else:                                          # rapid dives
                        base = X[i] if abs(E) >= 0.5 else X.mean(0)
                        Y = np.clip(rabbit - E * np.abs(J * rabbit - base), 0, 1)
                        fY = b(Y)
                        if fY < F[i]:
                            X[i], F[i] = Y, fY
                        else:
                            Z = np.clip(Y + rng.random(d) * _levy(rng, d), 0, 1)
                            fZ = b(Z)
                            if fZ < F[i]:
                                X[i], F[i] = Z, fZ
                if F[i] < rabbit_f:
                    rabbit, rabbit_f = X[i].copy(), F[i]
        # spend any budget left over by perturbing around the rabbit
        while True:
            Y = np.clip(rabbit + 0.05 * rng.normal(size=d), 0, 1)
            fY = b(Y)
            if fY < rabbit_f:
                rabbit, rabbit_f = Y, fY
    except Budget.Exhausted:
        pass
    return _result(b, "HHO")


ALGORITHMS = {"PSO": pso, "GA": ga, "HHO": hho}
