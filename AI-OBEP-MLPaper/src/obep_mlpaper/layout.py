"""Movable-wall (MW) problem of Lyu et al. (2025), Sec. 2 and 4.2, extended with
zone cooling setpoints bounded by the Malaysian 24 degC policy floor.

Decision vector x = [x_1 .. x_{n_r-1} | T_set,1 .. T_set,n_z]
  * inner wall positions [m] (paper's decision variables), and
  * the cooling setpoint of each heating/cooling zone, 24 <= T_set <= 27 degC
    (Malaysian government offices: air-conditioning no lower than 24 degC).

Room updating rules (paper Algorithms 3-5): area from walls, occupants
re-seated, room T / RH = overlap-weighted mean of the zone setpoints / RH.

Objectives (both minimised)
  f_cost : total HVAC power of all rooms [kW] = sum_j q_j * A_j * phi(T_j)
           q_j  = surrogate HVAC power density [W/m2] (XGBoost tuned by PSO/GA/HHO)
           phi  = 1 - BETA * (T_j - T_REF): setpoint correction (see below)
  f_tc   : occupant-weighted mean |PMV| (ISO 7730), paper Eq. (11)

Why the setpoint correction: in ROBOD office-1 the logged setpoint is constant
(25 degC), so the data cannot identify how energy responds to a setpoint
change; the measured indoor temperature is an *outcome* that rises with load
(it correlates positively with energy). The surrogate is therefore queried at
the training reference temperature T_REF and the setpoint effect is applied
with BETA = 9.6 %/degC, from Hoyt, Arens & Zhang (2015, Build. Environ. 88:89-96):
raising the cooling setpoint 22.2 -> 25 degC saved 27 % of HVAC energy.
"""
from dataclasses import dataclass, field

import numpy as np

from .comfort import pmv
from .data import FEATURES, ROOM_AREA

OFFICE, MEETING = 1, 2
PENALTY_T, PENALTY_RH = 32.0, 100.0      # Algorithm 5 robustness penalty
T_MIN, T_MAX = 24.0, 27.0                # setpoint bounds [degC]; 24 = Malaysian floor
T_REF = 27.4                             # median operating-hour indoor T of office-1 [degC]
BETA = 0.27 / 2.8                        # fractional HVAC energy change per degC
EVAL_HOUR = 14.5                         # snapshot time (paper's midday outdoor conditions)


@dataclass
class Scenario:
    name: str
    B: float
    init_lengths: list
    room_types: list
    init_occupants: list
    zone_areas: list
    zone_rh: list
    r_area_min: float
    o_area_min: float
    zone_t_paper: list = None            # paper's fixed zone temperatures (reference only)
    d_out: dict = field(default_factory=lambda: dict(
        dry_bulb_temp=27.5, outdoor_relative_humidity=85.0,
        wind_speed=1.0, global_horizontal_solar_radiation=650.0))

    def __post_init__(self):
        self.n_rooms = len(self.init_lengths)
        self.n_walls = self.n_rooms - 1
        self.n_zones = len(self.zone_areas)
        self.L = float(np.sum(self.init_lengths))
        self.x_init = np.concatenate([[0.0], np.cumsum(self.init_lengths)])
        self.zone_edges = np.concatenate([[0.0], np.cumsum(self.zone_areas) / self.B])
        assert abs(self.zone_edges[-1] - self.L) < 1e-6
        self.o_init = np.asarray(self.init_occupants, int)
        self.is_office = np.asarray(self.room_types) == OFFICE
        self.xl = np.r_[np.zeros(self.n_walls), np.full(self.n_zones, T_MIN)]
        self.xu = np.r_[np.full(self.n_walls, self.L), np.full(self.n_zones, T_MAX)]

    def baseline(self, setpoint=T_MIN):
        """Existing layout operated at the policy setpoint (default 24 degC)."""
        return np.r_[self.x_init[1:-1], np.full(self.n_zones, setpoint)]


def case_study_1() -> Scenario:
    """Paper Table 2: 5 offices (3 people) + 1 meeting room, 20 m2 each, B = 4 m."""
    return Scenario(name="case1_simple", B=4.0, init_lengths=[5.0] * 6,
                    room_types=[OFFICE] * 5 + [MEETING], init_occupants=[3, 3, 3, 3, 3, 0],
                    zone_areas=[20.0] * 6, zone_rh=[85.0, 82.0, 80.0, 86.0, 89.0, 85.0],
                    r_area_min=6.0, o_area_min=6.0,
                    zone_t_paper=[22.5, 23.0, 23.5, 22.9, 22.2, 22.0])


def case_study_2() -> Scenario:
    """Paper Table 5: 8 offices of the Watson Building, B = 3 m."""
    areas = [19.0, 19.0, 19.0, 19.0, 38.4, 19.0, 20.9, 12.6]
    return Scenario(name="case2_watson", B=3.0, init_lengths=[a / 3.0 for a in areas],
                    room_types=[OFFICE] * 8, init_occupants=[1, 1, 1, 1, 5, 1, 3, 1],
                    zone_areas=areas, zone_rh=[60.1, 60.0, 71.3, 74.0, 72.0, 57.5, 61.0, 58.8],
                    r_area_min=7.3, o_area_min=6.0,
                    zone_t_paper=[28.0, 27.9, 27.1, 27.5, 27.2, 26.9, 27.1, 26.7])


# ---- room updating rules ---------------------------------------------------
def split_x(sc: Scenario, X):
    X = np.atleast_2d(np.asarray(X, float))
    walls = np.sort(np.clip(X[:, :sc.n_walls], 0.0, sc.L), axis=1)
    sets = np.clip(X[:, sc.n_walls:], T_MIN, T_MAX)
    n = len(X)
    W = np.hstack([np.zeros((n, 1)), walls, np.full((n, 1), sc.L)])
    return W, sets


def update_room_area(sc, W):
    """Algorithm 3."""
    return np.maximum(np.diff(W, axis=1) * sc.B, 0.0)


def update_room_occupants(sc, area):
    """Algorithm 4: displaced occupants are re-seated in offices with spare capacity."""
    N, nr = area.shape
    cap = np.where(sc.is_office, np.floor(area / sc.o_area_min + 1e-9), 0).astype(int)
    occ = np.tile(sc.o_init, (N, 1))
    over = np.maximum(sc.o_init - cap, 0)
    occ = occ - over
    rem = over.sum(axis=1)
    free = cap - sc.o_init
    for j in range(nr):
        take = np.where((free[:, j] >= 1) & (rem > 0), np.minimum(free[:, j], rem), 0)
        occ[:, j] += take
        rem = rem - take
    return occ, rem


def overlap_weights(sc, W):
    lo, hi = W[:, :-1, None], W[:, 1:, None]
    ze_lo, ze_hi = sc.zone_edges[None, None, :-1], sc.zone_edges[None, None, 1:]
    overlap = np.clip(np.minimum(hi, ze_hi) - np.maximum(lo, ze_lo), 0, None)
    length = (hi - lo)[..., 0]
    ok = length > 1e-9
    return overlap / np.where(ok, length, 1.0)[..., None], ok


def update_room_t_rh(sc, W, setpoints):
    """Algorithm 5 with zone temperatures = zone setpoints (decision variables)."""
    w, ok = overlap_weights(sc, W)
    t = np.einsum("nrz,nz->nr", w, setpoints)
    rh = w @ np.asarray(sc.zone_rh)
    return np.where(ok, t, PENALTY_T), np.where(ok, rh, PENALTY_RH)


# ---- evaluation ------------------------------------------------------------
def surrogate_inputs(sc, rh, occ, area):
    """Feature matrix for the surrogate, one row per room.

    Occupants are expressed as the office-1-equivalent count (same people per m2
    as in the 98.4 m2 training room), so the density-driven load transfers to
    rooms of any size; the predicted W/m2 is then multiplied by the room area.
    """
    N, nr = rh.shape
    occ_eq = occ * ROOM_AREA / np.maximum(area, 1e-9)
    h = EVAL_HOUR
    vals = {"air_temperature": np.full((N, nr), T_REF), "indoor_relative_humidity": rh,
            "occupant_count": occ_eq, "hour_sin": np.full((N, nr), np.sin(2 * np.pi * h / 24)),
            "hour_cos": np.full((N, nr), np.cos(2 * np.pi * h / 24)), "hvac_schedule": np.ones((N, nr))}
    vals.update({k: np.full((N, nr), v) for k, v in sc.d_out.items()})
    return np.stack([vals[f] for f in FEATURES], axis=-1).reshape(-1, len(FEATURES))


def evaluate(sc: Scenario, surrogate, X) -> dict:
    """Algorithm 2: decision vectors -> objectives, constraints and room state."""
    W, sets = split_x(sc, X)
    area = update_room_area(sc, W)
    occ, unassigned = update_room_occupants(sc, area)
    T, RH = update_room_t_rh(sc, W, sets)
    N, nr = area.shape

    q = surrogate.predict(surrogate_inputs(sc, RH, occ, area)).reshape(N, nr)   # W/m2
    phi = np.clip(1.0 - BETA * (T - T_REF), 0.0, None)
    kw = q * area / 1000.0 * phi
    f_cost = kw.sum(axis=1)

    p = pmv(T.ravel(), RH.ravel()).reshape(N, nr)
    f_tc = (np.abs(p) * occ).sum(axis=1) / np.maximum(occ.sum(axis=1), 1)

    G = np.hstack([sc.r_area_min - area, unassigned[:, None].astype(float)])
    return dict(F=np.column_stack([f_cost, f_tc]), G=G, W=W, setpoints=sets, area=area,
                occ=occ, T=T, RH=RH, pmv=p, kw=kw)
