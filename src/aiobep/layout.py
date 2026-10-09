"""Movable-wall (MW) problem of the ML paper (Lyu et al., 2025), Sec. 2 and 4.2.

A floor is a 1-D row of ``n_r`` rooms separated by ``n_r + 1`` walls at x-positions
``x_0=0 < x_1 < ... < x_nr = L``; the room depth is the constant ``B``. The decision
variables are the inner wall positions. A candidate layout is turned into room
area / occupants / temperature / humidity by the three *room updating rules*
(Algorithms 3-5), then scored with two objectives:

  f_cost : total HVAC energy of all rooms predicted by the surrogate  [kW]
  f_tc   : occupant-weighted mean |PMV|                               [-]
"""
from dataclasses import dataclass, field

import numpy as np

from .comfort import pmv
from .data import FEATURES

OFFICE, MEETING, UNUSED = 1, 2, 0
PENALTY_T, PENALTY_RH = 32.0, 100.0  # Algorithm 5 robustness penalty


@dataclass
class Scenario:
    name: str
    B: float                       # room depth [m]
    init_lengths: list             # initial room lengths [m]  (rooms left -> right)
    room_types: list               # OFFICE / MEETING / UNUSED per room
    init_occupants: list           # initial occupants per room
    zone_areas: list               # heating-zone areas [m2]; sum must equal L*B
    zone_t: list                   # zone target temperature [degC]
    zone_rh: list                  # zone target relative humidity [%]
    r_area_min: float              # minimum room area [m2]
    o_area_min: float              # minimum area per occupant [m2]
    d_out: dict = field(default_factory=lambda: dict(
        dry_bulb_temp=27.5, outdoor_relative_humidity=85.0,
        wind_speed=1.0, global_horizontal_solar_radiation=650.0))
    movable: list = None           # bool per inner wall (default: all movable)

    def __post_init__(self):
        self.n_rooms = len(self.init_lengths)
        self.L = float(np.sum(self.init_lengths))
        self.x_init = np.concatenate([[0.0], np.cumsum(self.init_lengths)])
        self.zone_edges = np.concatenate([[0.0], np.cumsum(self.zone_areas) / self.B])
        assert abs(self.zone_edges[-1] - self.L) < 1e-6, "zone areas must sum to L*B"
        assert len(self.room_types) == len(self.init_occupants) == self.n_rooms
        if self.movable is None:
            self.movable = [True] * (self.n_rooms - 1)
        self.o_init = np.asarray(self.init_occupants, int)
        self.is_office = np.asarray(self.room_types) == OFFICE


def case_study_1() -> Scenario:
    """Simple scenario (paper Table 2): 5 offices of 3 people + 1 meeting room, 20 m2 each."""
    return Scenario(
        name="case1_simple", B=4.0, init_lengths=[5.0] * 6,
        room_types=[OFFICE] * 5 + [MEETING], init_occupants=[3, 3, 3, 3, 3, 0],
        zone_areas=[20.0] * 6, zone_t=[22.5, 23.0, 23.5, 22.9, 22.2, 22.0],
        zone_rh=[85.0, 82.0, 80.0, 86.0, 89.0, 85.0], r_area_min=6.0, o_area_min=6.0)


def case_study_2() -> Scenario:
    """Real-world-like scenario (paper Table 5): 8 offices of the Watson Building."""
    areas = [19.0, 19.0, 19.0, 19.0, 38.4, 19.0, 20.9, 12.6]
    B = 3.0
    return Scenario(
        name="case2_watson", B=B, init_lengths=[a / B for a in areas],
        room_types=[OFFICE] * 8, init_occupants=[1, 1, 1, 1, 5, 1, 3, 1],
        zone_areas=areas, zone_t=[28.0, 27.9, 27.1, 27.5, 27.2, 26.9, 27.1, 26.7],
        zone_rh=[60.1, 60.0, 71.3, 74.0, 72.0, 57.5, 61.0, 58.8],
        r_area_min=7.3, o_area_min=6.0)


# ---- room updating rules ---------------------------------------------------

def walls_from_inner(sc: Scenario, X_inner: np.ndarray) -> np.ndarray:
    """(N, n_r-1) inner wall positions -> (N, n_r+1) full walls; immovable ones
    keep their initial position, then rows are sorted (wall re-ordering)."""
    X = np.atleast_2d(X_inner).astype(float).copy()
    mov = np.asarray(sc.movable)
    X[:, ~mov] = sc.x_init[1:-1][~mov]
    X = np.sort(np.clip(X, 0.0, sc.L), axis=1)
    n = len(X)
    return np.hstack([np.zeros((n, 1)), X, np.full((n, 1), sc.L)])


def update_room_area(sc: Scenario, W: np.ndarray) -> np.ndarray:
    """Rule 1 / Algorithm 3: area_j = (x_{j+1}-x_j) * B (negative -> 0)."""
    return np.maximum(np.diff(W, axis=1) * sc.B, 0.0)


def update_room_occupants(sc: Scenario, area: np.ndarray):
    """Rule 2 / Algorithm 4, applied per solution.

    Offices keep their initial occupants; those that no longer fit
    (``floor(area/o_area_min)`` seats) are re-seated in offices with spare room.
    Returns (occupants (N, n_r), unassigned (N,)); ``unassigned > 0`` = infeasible.
    """
    N, nr = area.shape
    cap = np.where(sc.is_office, np.floor(area / sc.o_area_min + 1e-9), 0).astype(int)
    occ = np.tile(sc.o_init, (N, 1))
    over = np.maximum(sc.o_init - cap, 0)          # displaced from each room
    occ = occ - over
    rem = over.sum(axis=1)
    free = cap - sc.o_init                          # spare seats w.r.t. initial occupancy
    for j in range(nr):
        take = np.where((free[:, j] >= 1) & (rem > 0), np.minimum(free[:, j], rem), 0)
        occ[:, j] += take
        rem = rem - take
    return occ, rem


def update_room_t_rh(sc: Scenario, W: np.ndarray):
    """Rule 3 / Algorithm 5: room T / RH = overlap-weighted mean of heating-zone targets."""
    lo, hi = W[:, :-1, None], W[:, 1:, None]                       # (N, nr, 1)
    ze_lo, ze_hi = sc.zone_edges[None, None, :-1], sc.zone_edges[None, None, 1:]
    overlap = np.clip(np.minimum(hi, ze_hi) - np.maximum(lo, ze_lo), 0, None)  # (N, nr, nz)
    length = (hi - lo)[..., 0]
    ok = length > 1e-9
    w = overlap / np.where(ok, length, 1.0)[..., None]
    t = w @ np.asarray(sc.zone_t)
    rh = w @ np.asarray(sc.zone_rh)
    return np.where(ok, t, PENALTY_T), np.where(ok, rh, PENALTY_RH)


# ---- evaluation ------------------------------------------------------------

def evaluate(sc: Scenario, surrogate, X_inner: np.ndarray) -> dict:
    """Score a batch of candidate layouts (Algorithm 2). Returns objectives,
    constraint violations and the intermediate room state."""
    W = walls_from_inner(sc, X_inner)
    area = update_room_area(sc, W)
    occ, unassigned = update_room_occupants(sc, area)
    T, RH = update_room_t_rh(sc, W)
    N, nr = area.shape

    feat = {"air_temperature": T, "indoor_relative_humidity": RH, "occupant_count": occ}
    cols = [feat[f] if f in feat else np.full((N, nr), sc.d_out[f]) for f in FEATURES]
    Xs = np.stack(cols, axis=-1).reshape(-1, len(FEATURES))
    wh_m2 = surrogate.predict(Xs).reshape(N, nr)                    # Wh/m2 per 5 min
    kw = wh_m2 * area / 1000.0 * 12.0                               # kW (5-min energy x12)
    f_cost = kw.sum(axis=1)

    p = pmv(T.ravel(), RH.ravel()).reshape(N, nr)
    tot = np.maximum(occ.sum(axis=1), 1)
    f_tc = (np.abs(p) * occ).sum(axis=1) / tot

    g = np.hstack([sc.r_area_min - area,                            # room area >= r_area_min
                   unassigned[:, None].astype(float)])              # every occupant seated
    return dict(F=np.column_stack([f_cost, f_tc]), G=g, W=W, area=area, occ=occ,
                T=T, RH=RH, pmv=p, kw=kw)
