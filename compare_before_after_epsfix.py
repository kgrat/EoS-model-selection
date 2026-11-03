"""
Old vs new (energy-density fix) comparison for the three nested-sampling runs.

Old physics is evaluated on the old posterior samples, new physics on the new samples.
Usage:
  python compare_before_after_epsfix.py [new_results_dir] [--ndraw N]
"""
import argparse
import importlib.util
import os
import pickle
import numpy as np
import jax.numpy as jnp
from dynesty.utils import resample_equal

import jax_physics_poly as new_poly
import jax_physics_css as new_css
import jax_physics_spectral as new_spec

HERE = os.path.dirname(os.path.realpath(__file__))
OLD_DIR = os.path.join(HERE, "results_before_epsfix")

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

old_mods = {m: load_module(f"old_{m}", os.path.join(OLD_DIR, "old_engines", f"jax_physics_{m}.py"))
            for m in ["poly", "css", "spectral"]}
new_mods = {"poly": new_poly, "css": new_css, "spectral": new_spec}
LABELS = {"poly": "Piecewise polytrope", "css": "Hybrid (CSS)", "spectral": "Spectral"}

def posterior_draws(res, n, seed=0):
    w = np.exp(res.logwt - res.logz[-1])
    eq = resample_equal(res.samples, w / w.sum(), rstate=np.random.default_rng(seed))
    idx = np.random.default_rng(seed + 1).choice(len(eq), size=min(n, len(eq)), replace=False)
    return eq[idx]

def observables(mod, params):
    m, r = mod.compute_curve_jax(jnp.array(params))
    m, r = np.asarray(m), np.asarray(r)
    ok = np.isfinite(m) & np.isfinite(r)
    m, r = m[ok], r[ok]
    if len(m) < 2:
        return np.nan, np.nan, 0
    k = int(np.argmax(m))
    m_st, r_st = m[:k+1], r[:k+1]           # stable branch
    r14 = np.interp(1.4, m_st, r_st) if m_st[0] < 1.4 < m_st[-1] else np.nan
    return m[k], r14, k

def max_cs2(model, params):
    """Max dP/deps (units c^2) up to the central pressure of the max-mass star (new physics only)."""
    mod = new_mods[model]
    m, _, k = observables(mod, params)
    if not np.isfinite(m):
        return np.nan
    if model == "poly":
        seg = mod.build_segments(jnp.array(params))
        Pc = float(mod.pressure_of_rho(jnp.logspace(17.3, 18.6, 50)[k], seg))
        f = lambda P: mod.cs2_of_P(P, seg)
    elif model == "css":
        tb = mod.build_tables(jnp.array(params))
        Pc = float(mod.P_of_eps(jnp.logspace(17.3, 18.6, 50)[k] * mod.c**2, tb))
        f = lambda P: mod.cs2_of_P(P, tb)
    else:
        tb = mod.build_tables(jnp.array(params))
        Pc = float(jnp.logspace(33.5, 36.5, 50)[k])
        f = lambda P: mod.cs2_of_P(P, tb)
    Ps = jnp.logspace(30, np.log10(Pc), 400)
    return float(np.nanmax(np.asarray(jnp.vectorize(f)(Ps))))

def summarize(x):
    x = np.asarray(x)[np.isfinite(x)]
    lo, med, hi = np.percentile(x, [5, 50, 95])
    return f"{med:.2f} [{lo:.2f}, {hi:.2f}]"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new_dir", nargs="?", default=os.path.join(HERE, "results_after_epsfix_nlive300"))
    ap.add_argument("--ndraw", type=int, default=1000)
    args = ap.parse_args()

    rows, lnz = [], {}
    for model in ["poly", "css", "spectral"]:
        out = {}
        for tag, d, mods in [("old", OLD_DIR, old_mods), ("new", args.new_dir, new_mods)]:
            res = pickle.load(open(os.path.join(d, f"jax_{model}_results.pkl"), "rb"))
            draws = posterior_draws(res, args.ndraw)
            obs = np.array([observables(mods[model], p)[:2] for p in draws])
            out[tag] = dict(lnz=res.logz[-1], err=res.logzerr[-1], nlive=res.nlive,
                            mmax=summarize(obs[:, 0]), r14=summarize(obs[:, 1]), draws=draws)
            lnz[(model, tag)] = res.logz[-1]
        cs = np.array([max_cs2(model, p) for p in out["new"]["draws"][:300]])
        out["acausal"] = np.mean(cs[np.isfinite(cs)] > 1.0)
        rows.append((model, out))

    print(f"\n{'Model':22s}{'':5s}{'M_max [Msun] med [90%]':>26s}{'R_1.4 [km] med [90%]':>26s}"
          f"{'ln Z':>16s}{'ln B vs poly':>14s}{'B vs poly':>11s}")
    for model, out in rows:
        for tag in ["old", "new"]:
            o = out[tag]
            dl = o["lnz"] - lnz[("poly", tag)]
            print(f"{LABELS[model]:22s}{tag:5s}{o['mmax']:>26s}{o['r14']:>26s}"
                  f"{o['lnz']:>9.2f} +/- {o['err']:.2f}{dl:>14.2f}{np.exp(dl):>11.3f}")
    print("\nFraction of new posterior draws (first 300) with c_s^2 > c^2 below the max-mass centre:")
    for model, out in rows:
        print(f"  {LABELS[model]:22s}{100*out['acausal']:.0f}%")

if __name__ == "__main__":
    main()
