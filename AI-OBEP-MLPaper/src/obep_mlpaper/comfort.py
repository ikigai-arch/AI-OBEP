"""PMV thermal comfort (ISO 7730:2005), inputs as in Lyu et al. Sec. 5.1:
indoor air speed 0.1 m/s, 1.0 met, 0.5 clo, mean radiant T = air T."""
import numpy as np
from pythermalcomfort.models import pmv_ppd_iso

VR, MET, CLO = 0.1, 1.0, 0.5


def pmv(t_air, rh, vr=VR, met=MET, clo=CLO):
    t = np.atleast_1d(np.asarray(t_air, float))
    rh = np.clip(np.atleast_1d(np.asarray(rh, float)), 0, 100)
    out = pmv_ppd_iso(tdb=t, tr=t, vr=vr, rh=rh, met=met, clo=clo, model="7730-2005",
                      limit_inputs=False, round_output=False).pmv
    return np.atleast_1d(np.asarray(out, float))
