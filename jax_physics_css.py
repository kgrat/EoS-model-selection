import jax
import jax.numpy as jnp
from functools import partial

# Enabling 64-bit precision for accurate TOV integration
jax.config.update("jax_enable_x64", True)

# constants
G = 6.67430e-11
c = 2.99792458e8
M_sun = 1.989e30
RHO_NUC = 2.7e17

CRUST_GAMMA = 1.58425
CRUST_K = 1.0557e5
RHO_STITCH = 0.5 * RHO_NUC
P_STITCH = CRUST_K * RHO_STITCH**CRUST_GAMMA

# hybrid CSS Logic
# Everything the TOV solver needs is epsilon(P). The phases, in pressure:
#   crust   P < P_STITCH         eps = rho c^2 + P/(Gamma_c - 1)              (a = 0)
#   hadron  P_STITCH <= P < P_t  eps = (1 + a_h) rho c^2 + P/(Gamma_h - 1)    (Read et al. 2009)
#   quark   P >= P_t             eps = eps_q + (P - P_t)/c_s^2,  eps_q = eps_h(P_t) + Delta_eps
# a_h makes eps continuous at P_STITCH. eps jumps by Delta_eps (latent heat) at P_t.

@jax.jit
def build_tables(params):
    """Derived constants of the hybrid EoS for one parameter vector."""
    # Unpack parameters
    log_p1, gamma_h, log_p_trans, de_ratio, c_sq = params

    P1 = 10**log_p1
    P_trans = 10**log_p_trans

    # 1. Hadronic Setup
    rho_ref = 1.85 * RHO_NUC
    K_h = P1 / (rho_ref**gamma_h)

    # 2. Continuity of eps where the crust hands over to the hadronic polytrope (at P_STITCH)
    eps_stitch = RHO_STITCH * c**2 + P_STITCH / (CRUST_GAMMA - 1)
    rho_h0 = (P_STITCH / K_h)**(1.0/gamma_h)
    a_h = eps_stitch / (rho_h0 * c**2) - 1 - P_STITCH / (rho_h0 * (gamma_h - 1) * c**2)

    # 3. Transition Points
    rho_trans_h = (P_trans / K_h)**(1.0/gamma_h)
    eps_trans_h = (1 + a_h) * rho_trans_h * c**2 + P_trans / (gamma_h - 1)
    eps_trans_q = eps_trans_h + (de_ratio * RHO_NUC * c**2)

    # Rest-mass density at the start of the quark phase: the baryon chemical potential
    # mu = (eps + P)/n is continuous across a first-order transition
    rho_trans_q = rho_trans_h * (eps_trans_q + P_trans) / (eps_trans_h + P_trans)

    return dict(K_h=K_h, gamma_h=gamma_h, a_h=a_h, P_trans=P_trans, c_sq=c_sq,
                rho_trans_h=rho_trans_h, eps_trans_h=eps_trans_h,
                eps_trans_q=eps_trans_q, rho_trans_q=rho_trans_q)

def eps_of_P(P, tb):
    """Total energy density epsilon(P) in J/m^3."""
    P = jnp.maximum(P, 0.0)
    rho_crust = (P / CRUST_K)**(1.0 / CRUST_GAMMA)
    eps_crust = rho_crust * c**2 + P / (CRUST_GAMMA - 1)
    rho_hadron = (P / tb['K_h'])**(1.0 / tb['gamma_h'])
    eps_hadron = (1 + tb['a_h']) * rho_hadron * c**2 + P / (tb['gamma_h'] - 1)
    eps_quark = tb['eps_trans_q'] + (P - tb['P_trans']) / tb['c_sq']

    return jnp.where(P < P_STITCH, eps_crust,
            jnp.where(P < tb['P_trans'], eps_hadron,
             eps_quark))

def rho_of_P(P, tb):
    """Rest-mass density (kg/m^3). In the quark phase from d(rho)/rho = d(eps)/(eps + P)."""
    P = jnp.maximum(P, 0.0)
    rho_crust = (P / CRUST_K)**(1.0 / CRUST_GAMMA)
    rho_hadron = (P / tb['K_h'])**(1.0 / tb['gamma_h'])
    s = tb['c_sq']
    eps_q = eps_of_P(P, tb)
    rho_quark = tb['rho_trans_q'] * (((1 + s) * eps_q + tb['P_trans'] - s * tb['eps_trans_q'])
                                     / (tb['eps_trans_q'] + tb['P_trans']))**(1.0 / (1 + s))

    return jnp.where(P < P_STITCH, rho_crust,
            jnp.where(P < tb['P_trans'], rho_hadron,
             rho_quark))

def cs2_of_P(P, tb):
    """Squared sound speed dP/deps in units of c^2."""
    P = jnp.maximum(P, 1e-30)
    eps = eps_of_P(P, tb)
    return jnp.where(P < P_STITCH, CRUST_GAMMA * P / (eps + P),
            jnp.where(P < tb['P_trans'], tb['gamma_h'] * P / (eps + P),
             tb['c_sq']))

def _invert_monotone(f, target, log_lo=20.0, log_hi=40.0, n_iter=80):
    """Bisection in log10(P) for f(P) = target, f non-decreasing (jumps map to the jump pressure)."""
    def body(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        above = f(10**mid) >= target
        return (jnp.where(above, lo, mid), jnp.where(above, mid, hi))
    lo, hi = jax.lax.fori_loop(0, n_iter, body, (log_lo, log_hi))
    return 10**(0.5 * (lo + hi))

def P_of_eps(eps, tb):
    return _invert_monotone(lambda P: eps_of_P(P, tb), eps)

@jax.jit
def get_eos_properties(rho, params):
    """
    Calculates Pressure (P) for a given rest-mass Density (rho) and CSS Params.
    Inside the latent-heat gap (rho_trans_h < rho < rho_trans_q) P = P_trans.
    """
    tb = build_tables(params)
    return jnp.vectorize(lambda r: _invert_monotone(lambda P: rho_of_P(P, tb), r))(rho)

@jax.jit
def get_inverse_eos(P, params):
    """Calculates rest-mass Density (rho) from Pressure (P)."""
    return rho_of_P(P, build_tables(params))

@jax.jit
def get_energy_density(P, params):
    return eps_of_P(P, build_tables(params))

# 3. CUSTOM JAX ODE SOLVER (RK4)
# Replaces scipy.integrate.solve_ivp.
# We use a fixed-step integrator because it compiles extremely well on JAX.

@jax.jit
def tov_equations(r, state, tb):
    m, p = state
    eps_m = eps_of_P(p, tb) / c**2  # energy density in mass units (kg/m^3)

    # Standard TOV Equations (with total energy density, not rest-mass density)
    term1 = (eps_m + p/c**2)
    term2 = (m + 4*jnp.pi*r**3*p/c**2)
    term3 = (1 - 2*G*m/(r*c**2))

    # Avoid singularity at r=0 or Event Horizon
    term3 = jnp.maximum(term3, 1e-9)

    dPdr = -G * term1 * term2 / (r**2 * term3)
    dMdr = 4 * jnp.pi * r**2 * eps_m

    # Stop evolution if Pressure is 0 (Surface reached)
    dPdr = jnp.where(p <= 0, 0.0, dPdr)
    dMdr = jnp.where(p <= 0, 0.0, dMdr)

    return jnp.array([dMdr, dPdr])

@jax.jit
def solve_one_star(central_eps_m, tb):
    """
    Integrates one star from center to surface.
    central_eps_m is the central energy density in mass units, eps_c / c^2 (kg/m^3).
    """
    P_c = P_of_eps(central_eps_m * c**2, tb)

    # Initial Conditions (small sphere at r0)
    r0 = 10.0 # meters
    m0 = (4.0/3.0) * jnp.pi * r0**3 * central_eps_m
    init_state = jnp.array([m0, P_c])

    # Integration Grid: 3000 steps of 10 meters (30 km max radius)
    dr = 10.0
    n_steps = 3000

    # The Scan Loop
    def step_fn(state, i):
        r = r0 + i * dr

        # RK4 Integration
        k1 = dr * tov_equations(r, state, tb)
        k2 = dr * tov_equations(r + 0.5*dr, state + 0.5*k1, tb)
        k3 = dr * tov_equations(r + 0.5*dr, state + 0.5*k2, tb)
        k4 = dr * tov_equations(r + dr, state + k3, tb)

        new_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6.0

        # Clamp pressure to 0 if it goes negative
        new_state = jnp.array([new_state[0], jnp.maximum(new_state[1], -1.0)])

        # Record pressure for finding surface later
        # 2Gm/(rc^2) >= 1 means the configuration is inside its own horizon: not a static star
        compactness = 2*G*new_state[0] / ((r + dr) * c**2)
        return new_state, (new_state[1], compactness)

    final_state, (pressure_history, compactness) = jax.lax.scan(step_fn, init_state, jnp.arange(n_steps))
    horizon = jnp.any(compactness >= 1.0)

    # Find Surface: First index where Pressure <= 0
    is_positive = pressure_history > 0
    surface_idx = jnp.sum(is_positive) # Counts how many steps P > 0

    # Final Mass and Radius
    # We take the state *at* the surface index
    M_final = final_state[0] # Mass accumulates, so final state is roughly correct
    R_final = r0 + surface_idx * dr

    return jnp.where(horizon, jnp.nan, M_final / M_sun), jnp.where(horizon, jnp.nan, R_final / 1000.0)

# Vetorized solver

@partial(jax.jit, static_argnames=['n_points'])
def compute_curve_jax(params, n_points=50):
    # Log-space central energy densities (eps_c / c^2, kg/m^3)
    eps_ms = jnp.logspace(17.3, 18.6, n_points)
    tb = build_tables(params)

    # vmap: Map the 'solve_one_star' function over the central energy densities
    masses, radii = jax.vmap(solve_one_star, in_axes=(0, None))(eps_ms, tb)

    return masses, radii

@partial(jax.jit, static_argnames=['n_points'])
def central_pressures(params, n_points=50):
    """Central pressure of each star on the compute_curve_jax grid."""
    tb = build_tables(params)
    return jnp.vectorize(lambda e: P_of_eps(e * c**2, tb))(jnp.logspace(17.3, 18.6, n_points))

@jax.jit
def max_cs2_up_to(params, P_max):
    """Max c_s^2/c^2 for P <= P_max. Hadronic c_s^2 grows with P, so P_STITCH- and P_trans- are included."""
    tb = build_tables(params)
    Ps = jnp.concatenate([jnp.logspace(30.0, jnp.log10(P_max), 400),
                          jnp.minimum(jnp.array([P_STITCH, tb['P_trans']]) * (1 - 1e-9), P_max)])
    return jnp.max(jnp.vectorize(lambda P: cs2_of_P(P, tb))(Ps))
