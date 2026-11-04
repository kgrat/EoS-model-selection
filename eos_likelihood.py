"""
Shared likelihood for the three nested-sampling runners.

ln L = ln KDE_GW(R_1.4) + ln KDE_J0030(R_1.4) + ln < KDE_J0740(R(M)) >_{M ~ J0740 masses}

with physical cuts (soft penalties below -1e10 so the sampler can find the boundary):
  * M_max < 2.01 Msun
  * c_s^2 > c^2 anywhere below the central pressure of the maximum-mass star
Radii are read off the stable branch only (dM/d(central density) > 0).
"""
import os
import numpy as np
from scipy.stats import gaussian_kde

script_dir = os.path.dirname(os.path.realpath(__file__))

M_MAX_CUT = 2.01
N_MASS_QUANTILES = 200

def _load(filename, skip):
    return np.loadtxt(os.path.join(script_dir, filename), skiprows=skip)

# GW170817 (EoS-insensitive): mass column 0, radius column 4
_gw = _load('EoS-insensitive_posterior_samples.dat', 1)
kde_ligo = gaussian_kde(_gw[(_gw[:, 0] > 1.35) & (_gw[:, 0] < 1.45), 4])

# NICER J0030: radius column 0, mass column 1
_j0030 = _load('J0030_data.dat', 2)
kde_j0030 = gaussian_kde(_j0030[(_j0030[:, 1] > 1.35) & (_j0030[:, 1] < 1.45), 0])

# NICER J0740: 1D radius KDE (same bandwidth gaussian_kde would use), tabulated on a fine grid
# because the 2.0-2.15 Msun window holds ~6e5 samples; plus the mass distribution in that window.
_j0740 = _load('J0740_data.dat', 2)
_win = (_j0740[:, 1] > 2.0) & (_j0740[:, 1] < 2.15)
_r740, _m740 = _j0740[_win, 0], _j0740[_win, 1]
_h740 = len(_r740)**(-1.0 / 5.0) * np.std(_r740, ddof=1)        # Scott's rule, as gaussian_kde
_dr = 0.002
_edges = np.arange(_r740.min() - 6 * _h740, _r740.max() + 6 * _h740 + _dr, _dr)
_counts, _ = np.histogram(_r740, bins=_edges)
_kx = np.arange(-6 * _h740, 6 * _h740 + _dr, _dr)
_kern = np.exp(-0.5 * (_kx / _h740)**2)
J0740_R_GRID = 0.5 * (_edges[1:] + _edges[:-1])
J0740_PDF = np.convolve(_counts, _kern / _kern.sum(), mode='same') / (len(_r740) * _dr)
J0740_MASSES = np.quantile(_m740, (np.arange(N_MASS_QUANTILES) + 0.5) / N_MASS_QUANTILES)

def stable_radius_at(masses, radii, targets):
    """
    R(M) on the stable branch. masses/radii are ordered by increasing central density.
    Stable runs are maximal stretches where M increases; the lowest-density run that
    brackets the target mass is used. Returns NaN where no stable star has that mass.
    """
    targets = np.atleast_1d(np.asarray(targets, dtype=float))
    out = np.full(targets.shape, np.nan)
    inc = np.diff(masses) > 0
    i = 0
    n = len(masses)
    while i < n - 1:
        if not inc[i]:
            i += 1
            continue
        j = i
        while j < n - 1 and inc[j]:
            j += 1
        m_run, r_run = masses[i:j+1], radii[i:j+1]
        todo = np.isnan(out) & (targets >= m_run[0]) & (targets <= m_run[-1])
        out[todo] = np.interp(targets[todo], m_run, r_run)
        i = j
    return out

def log_like_j0740(masses, radii):
    """ln of the J0740 radius KDE averaged over the pulsar's mass samples (0 where M > M_max)."""
    R = stable_radius_at(masses, radii, J0740_MASSES)
    p = np.where(np.isfinite(R), np.interp(R, J0740_R_GRID, J0740_PDF, left=0.0, right=0.0), 0.0)
    L = np.mean(p)
    return np.log(L) if L > 0 else -1e10

def score_curve(masses, radii, central_P, max_cs2_up_to):
    """
    masses, radii, central_P: one M-R curve ordered by increasing central density.
    max_cs2_up_to(P): maximum c_s^2/c^2 of the EoS for pressures up to P.
    """
    masses, radii, central_P = (np.asarray(x) for x in (masses, radii, central_P))
    # Drop stars where the EoS overflowed or the integration crossed a horizon
    ok = np.isfinite(masses) & np.isfinite(radii)
    masses, radii, central_P = masses[ok], radii[ok], central_P[ok]
    if len(masses) < 5:
        return -1e10

    k = int(np.argmax(masses))
    M_max = masses[k]
    if M_max < M_MAX_CUT:
        return -1e10 - 1e4 * (M_MAX_CUT - M_max)**2

    cs2 = float(max_cs2_up_to(central_P[k]))
    if not np.isfinite(cs2) or cs2 > 1.0:
        return -1e10 - 1e4 * (min(cs2, 10.0) - 1.0)**2 if np.isfinite(cs2) else -1e10 - 1e5

    r_14 = stable_radius_at(masses, radii, 1.4)[0]
    if not np.isfinite(r_14):
        return -1e10

    score = kde_ligo.logpdf(r_14)[0] + kde_j0030.logpdf(r_14)[0]
    score += log_like_j0740(masses, radii)
    return score
