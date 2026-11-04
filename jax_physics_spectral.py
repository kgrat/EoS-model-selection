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
P_STITCH = CRUST_K * RHO_STITCH**CRUST_GAMMA
# Energy density of the crust polytrope at the stitch (a = 0): rho c^2 + P/(Gamma - 1)
EPS_STITCH = RHO_STITCH * c**2 + P_STITCH / (CRUST_GAMMA - 1)

# Tabulation grid for the Lindblom integrals: x = log(P / P_STITCH) from 0 to log(1e37 Pa / P_STITCH)
N_TABLE = 4000
X_MAX = jnp.log(1e37 / P_STITCH)

# 2. Spectral Parameterization (Lindblom 2010)
# Gamma(P) = exp( sum( gamma_i * log(p/p0)^i ) )

@jax.jit
def get_gamma(P, params):
    """
    Calculates the Adiabatic Index Gamma(P) from spectral params.
    Params: [gamma0, gamma1, gamma2, gamma3]
    """
    g0, g1, g2, g3 = params

    # Normalized pressure x = log(P / P_stitch)
    # We clip P to P_stitch to avoid log errors below the crust match
    x = jnp.log(jnp.maximum(P, P_STITCH) / P_STITCH)

    # Polynomial expansion for Gamma
    gamma_val = jnp.exp(g0 + g1*x + g2*x**2 + g3*x**3)

    # Physical stability check (Gamma must be > 0, usually > 1)
    # We let the solver handle crazy values naturally (star will collapse)
    return gamma_val

def _cumtrapz(y, dx):
    return jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (y[1:] + y[:-1]) * dx)])

@jax.jit
def build_tables(params):
    """
    Lindblom (2010) Eqs. (8)-(9), with x = log(P/P0):
      mu(x)  = exp( -int_0^x dx'/Gamma(x') )               (mu = rho0 / rho)
      eps(x) = eps0/mu(x) + (1/mu(x)) int_0^x mu(x') P0 e^x' / Gamma(x') dx'
      rho(x) = rho0 / mu(x)
    evaluated once per parameter vector by cumulative trapezoid on N_TABLE points.
    """
    g0, g1, g2, g3 = params
    x = jnp.linspace(0.0, X_MAX, N_TABLE)
    dx = x[1] - x[0]
    log_gamma = g0 + g1*x + g2*x**2 + g3*x**3

    # Kept in log space: for parameters where Gamma -> 0 at high P, mu underflows and
    # eps, rho overflow. The table stays finite; stars reaching that region come out non-finite.
    I = _cumtrapz(jnp.exp(-log_gamma), dx)
    log_eps = I + jnp.log(EPS_STITCH + _cumtrapz(P_STITCH * jnp.exp(x - I - log_gamma), dx))
    log_rho = jnp.log(RHO_STITCH) + I

    return x, log_eps, log_rho, jnp.exp(log_gamma)

def eps_of_P(P, tb):
    """Total energy density epsilon(P) in J/m^3."""
    x_tab, log_eps_tab, _, _ = tb
    P = jnp.maximum(P, 0.0)
    rho_crust = (P / CRUST_K)**(1.0 / CRUST_GAMMA)
    eps_crust = rho_crust * c**2 + P / (CRUST_GAMMA - 1)
    x = jnp.log(jnp.maximum(P, P_STITCH) / P_STITCH)
    eps_core = jnp.exp(jnp.interp(x, x_tab, log_eps_tab))
    return jnp.where(P < P_STITCH, eps_crust, eps_core)

def rho_of_P(P, tb):
    x_tab, _, log_rho_tab, _ = tb
    P = jnp.maximum(P, 0.0)
    rho_crust = (P / CRUST_K)**(1.0 / CRUST_GAMMA)
    x = jnp.log(jnp.maximum(P, P_STITCH) / P_STITCH)
    return jnp.where(P < P_STITCH, rho_crust, jnp.exp(jnp.interp(x, x_tab, log_rho_tab)))

def cs2_of_P(P, tb):
    """Squared sound speed dP/deps = Gamma P / (eps + P), in units of c^2."""
    x_tab, _, _, gamma_tab = tb
    x = jnp.log(jnp.maximum(P, P_STITCH) / P_STITCH)
    gamma = jnp.where(P < P_STITCH, CRUST_GAMMA, jnp.interp(x, x_tab, gamma_tab))
    return gamma * P / (eps_of_P(P, tb) + P)

# TOV solver
# epsilon(P) comes from the Lindblom table, so only M and P are integrated.

@jax.jit
def tov_equations(r, state, tb):
    m, p = state

    # TOV Factors (GR), with the total energy density in mass units
    eps_m = eps_of_P(p, tb) / c**2

    term1 = (eps_m + p/c**2)
    term2 = (m + 4*jnp.pi*r**3*p/c**2)
    term3 = jnp.maximum(1 - 2*G*m/(r*c**2), 1e-9)

    # Standard TOV Gradients
    dPdr = -G * term1 * term2 / (r**2 * term3)
    dMdr = 4 * jnp.pi * r**2 * eps_m

    # Stop evolution at surface (P=0)
    is_surface = p <= 0
    dPdr = jnp.where(is_surface, 0.0, dPdr)
    dMdr = jnp.where(is_surface, 0.0, dMdr)

    return jnp.array([dMdr, dPdr])

@jax.jit
def solve_one_star(P_central, tb):
    eps_c = eps_of_P(P_central, tb)

    r0 = 10.0
    m0 = (4.0/3.0) * jnp.pi * r0**3 * (eps_c / c**2)
    init_state = jnp.array([m0, P_central])

    dr = 10.0
    n_steps = 3000

    def step_fn(state, i):
        r = r0 + i * dr
        k1 = dr * tov_equations(r, state, tb)
        k2 = dr * tov_equations(r + 0.5*dr, state + 0.5*k1, tb)
        k3 = dr * tov_equations(r + 0.5*dr, state + 0.5*k2, tb)
        k4 = dr * tov_equations(r + dr, state + k3, tb)

        new_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6.0

        # Clamp P to 0 if negative
        new_state = jnp.array([new_state[0], jnp.maximum(new_state[1], -1.0)])

        # 2Gm/(rc^2) >= 1 means the configuration is inside its own horizon: not a static star
        compactness = 2*G*new_state[0] / ((r + dr) * c**2)
        return new_state, (new_state[1], compactness)

    final_state, (pressure_history, compactness) = jax.lax.scan(step_fn, init_state, jnp.arange(n_steps))
    horizon = jnp.any(compactness >= 1.0)
    surface_idx = jnp.sum(pressure_history > 0)

    M, R = final_state[0]/M_sun, (r0 + surface_idx*dr)/1000.0
    return jnp.where(horizon, jnp.nan, M), jnp.where(horizon, jnp.nan, R)

@partial(jax.jit, static_argnames=['n_points'])
def compute_curve_jax(params, n_points=50):
    # For Spectral, we sweep central Pressure: 10^33.5 to 10^36.5 Pa
    ps = jnp.logspace(33.5, 36.5, n_points)
    tb = build_tables(params)

    masses, radii = jax.vmap(solve_one_star, in_axes=(0, None))(ps, tb)
    return masses, radii

@partial(jax.jit, static_argnames=['n_points'])
def central_pressures(params, n_points=50):
    """Central pressure of each star on the compute_curve_jax grid."""
    return jnp.logspace(33.5, 36.5, n_points)

@jax.jit
def max_cs2_up_to(params, P_max):
    """Max c_s^2/c^2 for P <= P_max (Gamma(P) is arbitrary here, so a dense grid is used)."""
    tb = build_tables(params)
    Ps = jnp.concatenate([jnp.logspace(30.0, jnp.log10(P_max), 1000), jnp.array([P_STITCH * (1 - 1e-9)])])
    Ps = jnp.minimum(Ps, P_max)
    return jnp.max(jnp.vectorize(lambda P: cs2_of_P(P, tb))(Ps))

@jax.jit
def get_eos_properties(rho_grid, params):
    """
    P(rho) for plotting, from the same Lindblom table (rho = rho0 / mu).
    """
    x_tab, _, log_rho_tab, _ = build_tables(params)
    P_core = P_STITCH * jnp.exp(jnp.interp(jnp.log(rho_grid), log_rho_tab, x_tab))

    # Crust equation
    P_crust = CRUST_K * rho_grid**CRUST_GAMMA

    # Stitch them together
    return jnp.where(rho_grid < RHO_STITCH, P_crust, P_core)

@jax.jit
def get_energy_density(P, params):
    return eps_of_P(P, build_tables(params))
