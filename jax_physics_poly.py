import jax
import jax.numpy as jnp
from functools import partial

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

# constants
G = 6.67430e-11
c = 2.99792458e8
M_sun = 1.989e30
RHO_NUC = 2.7e17

# Crust constants
CRUST_K = 1.0557e5
CRUST_GAMMA = 1.58425
RHO_STITCH = 0.5 * RHO_NUC

# Standard polytrope eos logic
# Params: [logP1, logP2, logP3]
# We reconstruct the piecewise polytrope from these 3 points + Fixed Densities
#
# Every segment i is stored as (rho_lo_i, P_lo_i, K_i, Gamma_i, a_i) with
#   P(rho)       = K_i rho^Gamma_i
#   epsilon(rho) = (1 + a_i) rho c^2 + K_i rho^Gamma_i / (Gamma_i - 1)
# and a_i fixed by continuity of epsilon at rho_lo_i (Read et al. 2009, Sec. II).
# rho_lo_0 = 0 and a_0 = 0 (the crust starts at zero density with no offset).

def segments_from_arrays(rho_lo, K, Gamma):
    """
    Builds the segment table from dividing densities and (K, Gamma) per segment.
    rho_lo[0] must be 0. Works for any number of segments (used by the SLy test).
    """
    rho_lo = jnp.asarray(rho_lo, dtype=jnp.float64)
    K = jnp.asarray(K, dtype=jnp.float64)
    Gamma = jnp.asarray(Gamma, dtype=jnp.float64)

    P_lo = K * rho_lo**Gamma

    # a_i from continuity of epsilon at each dividing density
    a = [jnp.array(0.0)]
    for i in range(1, rho_lo.shape[0]):
        r = rho_lo[i]
        eps_prev = (1 + a[i-1]) * r * c**2 + K[i-1] * r**Gamma[i-1] / (Gamma[i-1] - 1)
        a.append(eps_prev / (r * c**2) - 1 - K[i] * r**(Gamma[i] - 1) / ((Gamma[i] - 1) * c**2))

    return rho_lo, P_lo, K, Gamma, jnp.stack(a)

@jax.jit
def build_segments(params):
    log_p1, log_p2, log_p3 = params

    P1 = 10**log_p1
    P2 = 10**log_p2
    P3 = 10**log_p3

    # Fixed Density Knots (Read et al 2009)
    rho_1 = 1.85 * RHO_NUC
    rho_2 = 3.70 * RHO_NUC
    rho_3 = 7.40 * RHO_NUC

    # Match Crust at Stitch
    P_stitch = CRUST_K * RHO_STITCH**CRUST_GAMMA

    # Calculate Slopes (Gammas)
    # Gamma = (logP_next - logP_prev) / (logRho_next - logRho_prev)
    g1 = (jnp.log(P1) - jnp.log(P_stitch)) / (jnp.log(rho_1) - jnp.log(RHO_STITCH))
    g2 = (jnp.log(P2) - jnp.log(P1)) / (jnp.log(rho_2) - jnp.log(rho_1))
    g3 = (jnp.log(P3) - jnp.log(P2)) / (jnp.log(rho_3) - jnp.log(rho_2))

    # Calculate Constants (Ks) -> P = K * rho^Gamma
    k1 = P1 / rho_1**g1
    k2 = P2 / rho_2**g2
    k3 = P3 / rho_3**g3

    # Segment 3 is the simple high-density extension above rho_2
    return segments_from_arrays(
        jnp.array([0.0, RHO_STITCH, rho_1, rho_2]),
        jnp.array([CRUST_K, k1, k2, k3]),
        jnp.array([CRUST_GAMMA, g1, g2, g3]))

# Segment lookups (index gather instead of nested jnp.where, so any number of segments works)
def _idx_from_rho(rho, seg):
    rho_lo = seg[0]
    return jnp.sum(rho >= rho_lo[1:])

def _idx_from_P(P, seg):
    P_lo = seg[1]
    return jnp.sum(P >= P_lo[1:])

def pressure_of_rho(rho, seg):
    _, _, K, Gamma, _ = seg
    i = _idx_from_rho(rho, seg)
    return K[i] * rho**Gamma[i]

def eps_of_rho(rho, seg):
    """Total energy density epsilon(rho) in J/m^3."""
    _, _, K, Gamma, a = seg
    i = _idx_from_rho(rho, seg)
    return (1 + a[i]) * rho * c**2 + K[i] * rho**Gamma[i] / (Gamma[i] - 1)

def rho_of_P(P, seg):
    _, _, K, Gamma, _ = seg
    P = jnp.maximum(P, 0.0)
    i = _idx_from_P(P, seg)
    return (P / K[i])**(1.0 / Gamma[i])

def eps_of_P(P, seg):
    """epsilon(P): P -> rho (inverse EOS) -> epsilon(rho), all on the same segment."""
    _, _, K, Gamma, a = seg
    P = jnp.maximum(P, 0.0)
    i = _idx_from_P(P, seg)
    rho = (P / K[i])**(1.0 / Gamma[i])
    return (1 + a[i]) * rho * c**2 + P / (Gamma[i] - 1)

def cs2_of_P(P, seg):
    """Squared sound speed in units of c^2: dP/deps = Gamma P / (eps + P)."""
    Gamma = seg[3]
    i = _idx_from_P(P, seg)
    return Gamma[i] * P / (eps_of_P(P, seg) + P)

# Public wrappers accept scalars or arrays (used by the plotting scripts)
@jax.jit
def get_eos_properties(rho, params):
    seg = build_segments(params)
    return jnp.vectorize(lambda x: pressure_of_rho(x, seg))(rho)

@jax.jit
def get_inverse_eos(P, params):
    seg = build_segments(params)
    return jnp.vectorize(lambda x: rho_of_P(x, seg))(P)

@jax.jit
def get_energy_density(rho, params):
    seg = build_segments(params)
    return jnp.vectorize(lambda x: eps_of_rho(x, seg))(rho)

# jax tov solver (RK4)
# The TOV equations need the total energy density epsilon, not rho:
#   dm/dr = 4 pi r^2 epsilon/c^2
#   dP/dr = -G (epsilon/c^2 + P/c^2)(m + 4 pi r^3 P/c^2) / (r^2 (1 - 2Gm/(rc^2)))
@jax.jit
def tov_equations(r, state, seg):
    m, p = state
    eps_m = eps_of_P(p, seg) / c**2  # energy density in mass units (kg/m^3)

    term1 = (eps_m + p/c**2)
    term2 = (m + 4*jnp.pi*r**3*p/c**2)
    term3 = jnp.maximum(1 - 2*G*m/(r*c**2), 1e-9)

    dPdr = -G * term1 * term2 / (r**2 * term3)
    dMdr = 4 * jnp.pi * r**2 * eps_m

    dPdr = jnp.where(p <= 0, 0.0, dPdr)
    dMdr = jnp.where(p <= 0, 0.0, dMdr)

    return jnp.array([dMdr, dPdr])

@jax.jit
def solve_one_star(central_rho, seg):
    P_c = pressure_of_rho(central_rho, seg)
    eps_c = eps_of_rho(central_rho, seg)

    r0 = 10.0
    m0 = (4.0/3.0) * jnp.pi * r0**3 * eps_c / c**2
    init_state = jnp.array([m0, P_c])

    dr = 10.0
    n_steps = 3000

    def step_fn(state, i):
        r = r0 + i * dr
        k1 = dr * tov_equations(r, state, seg)
        k2 = dr * tov_equations(r + 0.5*dr, state + 0.5*k1, seg)
        k3 = dr * tov_equations(r + 0.5*dr, state + 0.5*k2, seg)
        k4 = dr * tov_equations(r + dr, state + k3, seg)

        new_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        new_state = jnp.array([new_state[0], jnp.maximum(new_state[1], -1.0)])
        # 2Gm/(rc^2) >= 1 means the configuration is inside its own horizon: not a static star
        compactness = 2*G*new_state[0] / ((r + dr) * c**2)
        return new_state, (new_state[1], compactness)

    final_state, (pressure_history, compactness) = jax.lax.scan(step_fn, init_state, jnp.arange(n_steps))
    horizon = jnp.any(compactness >= 1.0)

    surface_idx = jnp.sum(pressure_history > 0)

    M, R = final_state[0]/M_sun, (r0 + surface_idx*dr)/1000.0
    return jnp.where(horizon, jnp.nan, M), jnp.where(horizon, jnp.nan, R)

@jax.jit
def compute_curve_from_segments(seg, rhos):
    return jax.vmap(solve_one_star, in_axes=(0, None))(rhos, seg)

# Vectorization
@partial(jax.jit, static_argnames=['n_points'])
def compute_curve_jax(params, n_points=50):
    rhos = jnp.logspace(17.3, 18.6, n_points)
    return compute_curve_from_segments(build_segments(params), rhos)
