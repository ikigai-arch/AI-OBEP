"""PMV-based thermal comfort (ISO 7730:2005), as in the ML paper (Sec. 2.3)."""
import numpy as np
from pythermalcomfort.models import pmv_ppd_iso

# Paper case study 1: v=0.1 m/s indoor air speed, 1 met, 0.5 clo.
DEFAULTS = dict(vr=0.1, met=1.0, clo=0.5)


def pmv(t_air, rh, vr=0.1, met=1.0, clo=0.5):
    """Vectorised PMV; mean radiant temperature is assumed equal to air temperature."""
    t = np.atleast_1d(np.asarray(t_air, float))
    rh = np.atleast_1d(np.asarray(rh, float))
    out = pmv_ppd_iso(tdb=t, tr=t, vr=vr, rh=np.clip(rh, 0, 100), met=met, clo=clo,
                      model="7730-2005", limit_inputs=False, round_output=False).pmv
    return np.atleast_1d(np.asarray(out, float))
