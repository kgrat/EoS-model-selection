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
EPS_STITCH = RHO_STITCH * c**2 # Approx energy density at stitch

# 2. Spctral Parameterization (Lindblom 2010)
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

# TOV solver
# We solve dM/dr, dP/dr, AND dEpsilon/dr simultaneously.

@jax.jit
def tov_equations(r, state, params):
    m, p, eps = state # Unpack 3 variables
    
    # 1. Determine Gamma
    # If P < P_stitch, we are in Crust (Polytrope Gamma)
    # If P >= P_stitch, we are in Core (Spectral Gamma)
    gamma = jnp.where(p < P_STITCH, CRUST_GAMMA, get_gamma(p, params))
    
    # 2. TOV Factors (GR)
    # eps is energy density (rho * c^2)
    rho = eps / c**2
    
    term1 = (rho + p/c**2)
    term2 = (m + 4*jnp.pi*r**3*p/c**2)
    term3 = jnp.maximum(1 - 2*G*m/(r*c**2), 1e-9)
    
    # Standard TOV Gradients
    dPdr = -G * term1 * term2 / (r**2 * term3)
    dMdr = 4 * jnp.pi * r**2 * rho
    
    # 3. The EoS Evolution Equation (Thermodynamics)
    # d(epsilon)/dr = (d(epsilon)/dP) * (dP/dr)
    # From thermodynamics: d(epsilon)/dP = (epsilon + P) / (P * Gamma)
    
    dEps_dP = (eps + p/c**2 * c**2) / (jnp.maximum(p, 1e-5) * gamma) 
    # d(eps)/dp = (eps + P) / (P * Gamma) is correct for relativistic enthalpy.
    
    dEpsdr = dEps_dP * dPdr
    
    # Stop evolution at surface (P=0)
    # Mask gradients if P <= 0
    is_surface = p <= 0
    dPdr = jnp.where(is_surface, 0.0, dPdr)
    dMdr = jnp.where(is_surface, 0.0, dMdr)
    dEpsdr = jnp.where(is_surface, 0.0, dEpsdr)
    
    return jnp.array([dMdr, dPdr, dEpsdr])

@jax.jit
def solve_one_star(central_rho, params):
    # Initial Conditions at center
    P_c_guess = CRUST_K * central_rho**CRUST_GAMMA # Rough guess, actually we iterate P_c?
   
    P_c = CRUST_K * central_rho**CRUST_GAMMA * 10.0 # Boosted guess for core
    
    
    P_central = central_rho * c**2 # Just a scaling factor to get P into 10^33 range
    # Re-map input range 10^17.3 -> 10^33 Pa roughly
    
    # We assume start of integration at P_stitch with known Eps_stitch,
    
    
    # we assume Gamma is constant near center for the first step.
    gamma_c = get_gamma(P_central, params)
    # P = K * eps^Gamma -> eps = (P/K)^(1/Gamma)
    # We estimate epsilon_c assuming it behaves like a polytrope locally
    # This is valid for the very center 1-meter sphere.
    eps_c = (P_central / P_STITCH)**(1.0/gamma_c) * EPS_STITCH
    
    r0 = 10.0
    m0 = (4.0/3.0) * jnp.pi * r0**3 * (eps_c / c**2)
    init_state = jnp.array([m0, P_central, eps_c])
    
    dr = 10.0
    n_steps = 3000
    
    def step_fn(state, i):
        r = r0 + i * dr
        k1 = dr * tov_equations(r, state, params)
        k2 = dr * tov_equations(r + 0.5*dr, state + 0.5*k1, params)
        k3 = dr * tov_equations(r + 0.5*dr, state + 0.5*k2, params)
        k4 = dr * tov_equations(r + dr, state + k3, params)
        
        new_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        
        # Clamp P and Eps to 0 if negative
        safe_p = jnp.maximum(new_state[1], -1.0)
        safe_eps = jnp.maximum(new_state[2], 0.0)
        new_state = jnp.array([new_state[0], safe_p, safe_eps])
        
        return new_state, new_state[1] 

    final_state, pressure_history = jax.lax.scan(step_fn, init_state, jnp.arange(n_steps))
    surface_idx = jnp.sum(pressure_history > 0)
    
    return final_state[0]/M_sun, (r0 + surface_idx*dr)/1000.0

@partial(jax.jit, static_argnames=['n_points'])
def compute_curve_jax(params, n_points=50):
    # For Spectral, we sweep Pressure, not Density
    # P range: 10^33 to 10^36 Pa
    ps = jnp.logspace(33.5, 36.5, n_points)
    
    # solve_one_star calculates P_central = input * c^2.
    # we need to inverse that scaling.
    inputs = ps / c**2 
    
    masses, radii = jax.vmap(solve_one_star, in_axes=(0, None))(inputs, params)
    return masses, radii