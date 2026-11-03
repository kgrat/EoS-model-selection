"""
Validation of the energy-density correction in the JAX TOV engines.

1. SLy piecewise-polytrope fit (Read et al. 2009, Tables II & III) through the
   JAX polytrope engine and through an independent scipy reference solver.
   Published (tabulated SLy): M_max = 2.049 M_sun, R_1.4 = 11.736 km.
2. Continuity of epsilon at every segment boundary.
3. epsilon >= rho c^2 and causality (c_s^2 = dP/deps <= c^2).

Run: python validate_energy_density.py
"""
import numpy as np
import jax.numpy as jnp
from scipy.integrate import solve_ivp

import jax_physics_poly as poly
import jax_physics_css as css
import jax_physics_spectral as spec

G, c, M_sun, RHO_NUC = poly.G, poly.c, poly.M_sun, poly.RHO_NUC
C_CGS2 = (2.99792458e10)**2

failures = []

def check(cond, msg):
    print(("  [ok]   " if cond else "  [FAIL] ") + msg)
    if not cond:
        failures.append(msg)

# ---------------------------------------------------------------------------
# SLy in SI units
# Read Table II gives K for p/c^2 in g/cm^3 (see the footnote in Sec. III),
# so K_SI = K * c_cgs^2 * 0.1 * (1e-3)^Gamma  [P in Pa, rho in kg/m^3].
# ---------------------------------------------------------------------------
def sly_segments():
    crust_K_cgs = np.array([6.80110e-09, 1.06186e-06, 5.32697e+01, 3.99874e-08])
    crust_G = np.array([1.58425, 1.28733, 0.62223, 1.35692])
    crust_rho = np.array([0.0, 2.44034e+07, 3.78358e+11, 2.62780e+12]) * 1e3
    crust_K = crust_K_cgs * C_CGS2 * 0.1 * (1e-3)**crust_G

    log_p1_SI = 34.384 - 1.0                      # dyn/cm^2 -> Pa
    g1, g2, g3 = 3.005, 2.988, 2.851
    rho1, rho2 = 10**17.7, 10**18.0               # 10^14.7, 10^15.0 g/cm^3
    k1 = 10**log_p1_SI / rho1**g1
    k2 = k1 * rho1**g1 / rho1**g2
    k3 = k2 * rho2**g2 / rho2**g3
    rho0 = (crust_K[3] / k1)**(1.0 / (g1 - crust_G[3]))   # crust-core intersection

    rho_lo = np.concatenate([crust_rho, [rho0, rho1, rho2]])
    K = np.concatenate([crust_K, [k1, k2, k3]])
    Gam = np.concatenate([crust_G, [g1, g2, g3]])
    return rho_lo, K, Gam

def sly_params_this_model():
    """SLy mapped onto this repo's (logP1, logP2, logP3) at 1.85, 3.7, 7.4 rho_nuc (Pa)."""
    rho_lo, K, Gam = sly_segments()
    seg = poly.segments_from_arrays(rho_lo, K, Gam)
    rhos = jnp.array([1.85, 3.70, 7.40]) * RHO_NUC
    return [float(jnp.log10(poly.pressure_of_rho(r, seg))) for r in rhos]

# ---------------------------------------------------------------------------
# Independent reference: numpy epsilon(P) + scipy RK45 at tight tolerance
# ---------------------------------------------------------------------------
class RefPiecewise:
    def __init__(self, rho_lo, K, Gam):
        self.rho_lo, self.K, self.G = rho_lo, K, Gam
        self.P_lo = K * rho_lo**Gam
        a = [0.0]
        for i in range(1, len(K)):
            r = rho_lo[i]
            e = (1 + a[-1]) * r * c**2 + K[i-1] * r**Gam[i-1] / (Gam[i-1] - 1)
            a.append(e / (r * c**2) - 1 - K[i] * r**(Gam[i] - 1) / ((Gam[i] - 1) * c**2))
        self.a = np.array(a)

    def P(self, rho):
        i = np.searchsorted(self.rho_lo, rho, side='right') - 1
        return self.K[i] * rho**self.G[i]

    def eps_rho(self, rho):
        i = np.searchsorted(self.rho_lo, rho, side='right') - 1
        return (1 + self.a[i]) * rho * c**2 + self.K[i] * rho**self.G[i] / (self.G[i] - 1)

    def eps_P(self, P):
        if P <= 0:
            return 0.0
        i = np.searchsorted(self.P_lo, P, side='right') - 1
        rho = (P / self.K[i])**(1 / self.G[i])
        return (1 + self.a[i]) * rho * c**2 + P / (self.G[i] - 1)

def ref_star(rho_c, eos):
    P_c, e_c = eos.P(rho_c), eos.eps_rho(rho_c)
    r0 = 1e-3
    m0 = 4/3 * np.pi * r0**3 * e_c / c**2
    def rhs(r, y):
        m, P = y
        if P <= 0:
            return [0.0, 0.0]
        e = eos.eps_P(P) / c**2
        dP = -G * (e + P/c**2) * (m + 4*np.pi*r**3*P/c**2) / (r**2 * (1 - 2*G*m/(r*c**2)))
        return [4*np.pi*r**2*e, dP]
    surf = lambda r, y: y[1] - 1e-12 * P_c
    surf.terminal = True
    sol = solve_ivp(rhs, [r0, 40e3], [m0, P_c], events=surf, rtol=1e-10, atol=[1e10, 1e-6*P_c],
                    method='DOP853')
    return sol.y[0, -1] / M_sun, sol.t[-1] / 1e3

def mmax_r14(masses, radii):
    masses, radii = np.asarray(masses), np.asarray(radii)
    k = int(np.argmax(masses))
    m_st, r_st = masses[:k+1], radii[:k+1]       # stable branch only
    return masses[k], float(np.interp(1.4, m_st, r_st))

# ---------------------------------------------------------------------------
print("=" * 72)
print("1. SLy (Read et al. 2009 Table III) through the corrected solvers")
print("=" * 72)
rho_lo, K, Gam = sly_segments()
seg = poly.segments_from_arrays(rho_lo, K, Gam)
print(f"  crust-core intersection rho_0 = {rho_lo[4]:.4e} kg/m^3 = {rho_lo[4]/RHO_NUC:.3f} rho_nuc")

rhos = jnp.logspace(17.3, 18.7, 300)
m_jax, r_jax = poly.compute_curve_from_segments(seg, rhos)
M_jax, R14_jax = mmax_r14(m_jax, r_jax)

ref = RefPiecewise(rho_lo, K, Gam)
rhos_ref = np.logspace(17.3, 18.7, 120)
mr_ref = np.array([ref_star(r, ref) for r in rhos_ref])
M_ref, R14_ref = mmax_r14(mr_ref[:, 0], mr_ref[:, 1])

print(f"  {'':28s}{'M_max [Msun]':>14s}{'R_1.4 [km]':>12s}")
print(f"  {'Read 2009 tabulated SLy':28s}{2.049:14.3f}{11.736:12.3f}")
print(f"  {'Read 2009 PP fit (implied)':28s}{2.049*1.0002:14.3f}{11.736*(1-0.0021):12.3f}")
print(f"  {'scipy reference (DOP853)':28s}{M_ref:14.4f}{R14_ref:12.3f}")
print(f"  {'JAX engine (RK4, dr=10 m)':28s}{M_jax:14.4f}{R14_jax:12.3f}")
check(abs(M_jax / 2.049 - 1) < 0.02, f"JAX M_max within 2% of 2.049 ({100*(M_jax/2.049-1):+.2f}%)")
check(abs(R14_jax / 11.736 - 1) < 0.02, f"JAX R_1.4 within 2% of 11.736 km ({100*(R14_jax/11.736-1):+.2f}%)")
check(abs(M_jax / M_ref - 1) < 2e-3, f"JAX vs reference M_max agree to 0.2% ({100*(M_jax/M_ref-1):+.3f}%)")
check(abs(R14_jax - R14_ref) < 0.03, f"JAX vs reference R_1.4 agree to 30 m ({1e3*(R14_jax-R14_ref):+.0f} m)")

# What the old (rho instead of epsilon) TOV gives for the same EOS, for scale
def ref_star_old(rho_c, eos):
    P_c = eos.P(rho_c)
    r0 = 1e-3
    def rhs(r, y):
        m, P = y
        if P <= 0:
            return [0.0, 0.0]
        i = np.searchsorted(eos.P_lo, P, side='right') - 1
        rho = (P / eos.K[i])**(1 / eos.G[i])
        dP = -G * (rho + P/c**2) * (m + 4*np.pi*r**3*P/c**2) / (r**2 * (1 - 2*G*m/(r*c**2)))
        return [4*np.pi*r**2*rho, dP]
    surf = lambda r, y: y[1] - 1e-12 * P_c
    surf.terminal = True
    sol = solve_ivp(rhs, [r0, 40e3], [4/3*np.pi*r0**3*rho_c, P_c], events=surf, rtol=1e-10,
                    atol=[1e10, 1e-6*P_c], method='DOP853')
    return sol.y[0, -1] / M_sun, sol.t[-1] / 1e3
mr_old = np.array([ref_star_old(r, ref) for r in rhos_ref])
M_old, R14_old = mmax_r14(mr_old[:, 0], mr_old[:, 1])
print(f"  {'old TOV (rho in place of eps)':28s}{M_old:14.4f}{R14_old:12.3f}   <- size of the bug")

# Same SLy core, but with this repo's single-segment crust and 3-knot parameterization
p_sly = sly_params_this_model()
m_rep, r_rep = poly.compute_curve_from_segments(poly.build_segments(jnp.array(p_sly)), rhos)
M_rep, R14_rep = mmax_r14(m_rep, r_rep)
print(f"  SLy mapped to this repo's params (logP in Pa) = [{p_sly[0]:.3f}, {p_sly[1]:.3f}, {p_sly[2]:.3f}]")
print(f"  {'repo crust + SLy knots':28s}{M_rep:14.4f}{R14_rep:12.3f}   <- crust model effect")

# ---------------------------------------------------------------------------
print("=" * 72)
print("2. Continuity of epsilon at segment boundaries")
print("=" * 72)
def boundary_table(name, seg_, p_tol=1e-10):
    # p_tol: tolerance for the P-space lookup used inside the ODE. Read's Table II
    # K values are rounded to 6 digits, so for SLy P itself is only continuous to ~1e-5
    # at the crust boundaries and eps(P) inherits that; the rho-space check is exact.
    rho_lo_, P_lo_, K_, G_, a_ = [np.asarray(x) for x in seg_]
    print(f"  {name}")
    print(f"    {'rho_i [kg/m^3]':>16s}{'eps_left [J/m^3]':>20s}{'eps_right [J/m^3]':>20s}{'rel diff':>11s}{'a_i':>12s}")
    worst = 0.0
    for i in range(1, len(K_)):
        r = rho_lo_[i]
        eL = (1 + a_[i-1]) * r * c**2 + K_[i-1] * r**G_[i-1] / (G_[i-1] - 1)
        eR = (1 + a_[i]) * r * c**2 + K_[i] * r**G_[i] / (G_[i] - 1)
        rel = abs(eR / eL - 1)
        worst = max(worst, rel)
        print(f"    {r:16.4e}{eL:20.10e}{eR:20.10e}{rel:11.1e}{a_[i]:12.3e}")
    check(worst < 1e-12, f"{name}: epsilon(rho) continuous at all boundaries (max rel diff {worst:.1e})")
    # also through the public eps_of_P (P-space lookup, as used inside the ODE)
    worst_P = 0.0
    for i in range(1, len(K_)):
        Pl = P_lo_[i]
        eL = float(poly.eps_of_P(Pl * (1 - 1e-12), seg_))
        eR = float(poly.eps_of_P(Pl * (1 + 1e-12), seg_))
        worst_P = max(worst_P, abs(eR / eL - 1))
    check(worst_P < p_tol, f"{name}: epsilon(P) continuous (max rel diff {worst_P:.1e}, tol {p_tol:.0e})")

boundary_table("SLy (7 segments)", seg, p_tol=1e-3)
for p in [p_sly, [33.79, 34.43, 35.56]]:
    boundary_table(f"repo polytrope, params {p}", poly.build_segments(jnp.array(p)))

def css_boundaries(params):
    P_stitch = css.P_STITCH
    P_t = 10**params[2]
    tb = css.build_tables(jnp.array(params))
    eL = float(css.eps_of_P(P_stitch * (1 - 1e-12), tb)); eR = float(css.eps_of_P(P_stitch * (1 + 1e-12), tb))
    print(f"  CSS {params}")
    print(f"    crust->hadron  at P={P_stitch:.4e}: eps {eL:.10e} -> {eR:.10e}  (rel {abs(eR/eL-1):.1e})")
    eh = float(css.eps_of_P(P_t * (1 - 1e-12), tb)); eq = float(css.eps_of_P(P_t * (1 + 1e-12), tb))
    jump = (eq - eh) / (RHO_NUC * c**2)
    print(f"    hadron->quark  at P={P_t:.4e}: eps {eh:.10e} -> {eq:.10e}  (jump = {jump:.4f} rho_nuc c^2, "
          f"param = {params[3]:.4f})")
    check(abs(eR / eL - 1) < 1e-10, f"CSS {params}: eps continuous crust->hadron")
    check(abs(jump - params[3]) < 1e-8, f"CSS {params}: latent-heat jump equals Delta eps parameter")

for p in [[34.3, 3.0, 34.6, 0.5, 0.6], [33.8, 2.5, 34.0, 1.2, 0.4]]:
    css_boundaries(p)

for p in [[0.9, 0.1, -0.05, 0.01], [1.3, -0.4, 0.1, -0.05]]:
    tb = spec.build_tables(jnp.array(p))
    Ps = spec.P_STITCH
    eL = float(spec.eps_of_P(Ps * (1 - 1e-12), tb)); eR = float(spec.eps_of_P(Ps * (1 + 1e-12), tb))
    print(f"  spectral {p}: crust->core at P={Ps:.4e}: eps {eL:.10e} -> {eR:.10e} (rel {abs(eR/eL-1):.1e})")
    check(abs(eR / eL - 1) < 1e-8, f"spectral {p}: eps continuous at stitch")

# Analytic check of the Lindblom table: constant Gamma (gamma_1..3 = 0) is a single polytrope
# stitched to the crust, so eps must match the closed form (1 + a) rho c^2 + P/(Gamma - 1).
for Gam in [2.0, 3.0]:
    tb = spec.build_tables(jnp.array([np.log(Gam), 0.0, 0.0, 0.0]))
    K1 = spec.P_STITCH / spec.RHO_STITCH**Gam
    seg_1 = poly.segments_from_arrays([0.0, spec.RHO_STITCH], [spec.CRUST_K, K1], [spec.CRUST_GAMMA, Gam])
    Ps = jnp.logspace(np.log10(spec.P_STITCH) + 0.01, 36.5, 200)
    e_tab = np.array([float(spec.eps_of_P(P, tb)) for P in Ps])
    e_ana = np.array([float(poly.eps_of_P(P, seg_1)) for P in Ps])
    r_tab = np.array([float(spec.rho_of_P(P, tb)) for P in Ps])
    r_ana = np.array([float(poly.rho_of_P(P, seg_1)) for P in Ps])
    err_e = np.max(np.abs(e_tab / e_ana - 1)); err_r = np.max(np.abs(r_tab / r_ana - 1))
    print(f"  spectral Gamma={Gam} constant vs analytic polytrope: max rel err eps {err_e:.1e}, rho {err_r:.1e}")
    check(err_e < 1e-5 and err_r < 1e-5, f"spectral Lindblom table matches analytic polytrope (Gamma={Gam})")

# ---------------------------------------------------------------------------
print("=" * 72)
print("3. epsilon >= rho c^2 and causality (up to the max-mass central pressure)")
print("=" * 72)
def scan(name, eps_fn, rho_fn, cs2_fn, P_cmax):
    Ps = jnp.logspace(20, np.log10(P_cmax), 4000)
    eps = np.asarray(eps_fn(Ps)); rho = np.asarray(rho_fn(Ps)); cs2 = np.asarray(cs2_fn(Ps))
    fin = np.isfinite(eps) & np.isfinite(rho) & (rho > 0)
    if not fin.all():
        print(f"  {name}: EOS overflows (Gamma -> 0) for P > {float(Ps[np.argmin(fin)]):.3e} Pa; "
              f"checking the finite part only")
        Ps, eps, rho, cs2 = Ps[fin], eps[fin], rho[fin], cs2[fin]
    ratio = eps / (rho * c**2)
    k = int(np.argmax(cs2))
    print(f"  {name}: min eps/(rho c^2) = {ratio.min():.6f}; max c_s^2/c^2 = {cs2.max():.4f} "
          f"at P = {float(Ps[k]):.3e} Pa")
    check(ratio.min() >= 1 - 1e-12, f"{name}: eps >= rho c^2 everywhere")
    if cs2.max() > 1:
        print(f"         c_s > c for P > {float(Ps[np.argmax(cs2 > 1)]):.3e} Pa "
              f"(central P of max-mass star = {P_cmax:.3e} Pa)")
    return cs2.max()

k_max = int(np.argmax(np.asarray(m_jax)))
Pc_max_sly = float(poly.pressure_of_rho(rhos[k_max], seg))
v2 = scan("SLy", lambda P: jnp.vectorize(lambda x: poly.eps_of_P(x, seg))(P),
          lambda P: jnp.vectorize(lambda x: poly.rho_of_P(x, seg))(P),
          lambda P: jnp.vectorize(lambda x: poly.cs2_of_P(x, seg))(P), Pc_max_sly)
print(f"         v_s,max = {np.sqrt(v2):.4f} c   (Read Table III: tab 0.989, fit +1.41% -> ~1.003)")

def max_mass_Pc(masses, Pcs):
    return float(np.asarray(Pcs)[int(np.argmax(np.asarray(masses)))])

for p in [p_sly, [33.79, 34.43, 35.56]]:
    sg = poly.build_segments(jnp.array(p))
    rh = jnp.logspace(17.3, 18.6, 50)
    m, _ = poly.compute_curve_from_segments(sg, rh)
    Pc = max_mass_Pc(m, [poly.pressure_of_rho(r, sg) for r in rh])
    scan(f"poly {[round(x, 3) for x in p]}", jnp.vectorize(lambda x: poly.eps_of_P(x, sg)),
         jnp.vectorize(lambda x: poly.rho_of_P(x, sg)), jnp.vectorize(lambda x: poly.cs2_of_P(x, sg)), Pc)

for p in [[34.3, 3.0, 34.6, 0.5, 0.6], [33.8, 2.5, 34.0, 1.2, 0.4]]:
    tb = css.build_tables(jnp.array(p))
    em = jnp.logspace(17.3, 18.6, 50)
    m, _ = css.compute_curve_jax(jnp.array(p))
    Pc = max_mass_Pc(m, [css.P_of_eps(e * c**2, tb) for e in em])
    scan(f"CSS {p}", jnp.vectorize(lambda x: css.eps_of_P(x, tb)),
         jnp.vectorize(lambda x: css.rho_of_P(x, tb)), jnp.vectorize(lambda x: css.cs2_of_P(x, tb)), Pc)

for p in [[0.9, 0.1, -0.05, 0.01], [1.3, -0.4, 0.1, -0.05]]:
    tb = spec.build_tables(jnp.array(p))
    m, _ = spec.compute_curve_jax(jnp.array(p))
    Pc = max_mass_Pc(m, jnp.logspace(33.5, 36.5, 50))
    scan(f"spectral {p}", jnp.vectorize(lambda x: spec.eps_of_P(x, tb)),
         jnp.vectorize(lambda x: spec.rho_of_P(x, tb)), jnp.vectorize(lambda x: spec.cs2_of_P(x, tb)), Pc)

if failures:
    print(f"\n{len(failures)} check(s) FAILED:")
    for f in failures:
        print("  -", f)
else:
    print("\nAll checks passed.")
