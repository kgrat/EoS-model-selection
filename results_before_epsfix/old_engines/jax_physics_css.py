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

# hybrid CSS Logic
@jax.jit
def get_eos_properties(rho, params):
    """
    Calculates Pressure (P) for a given Density (rho) and CSS Params.
    This replaces 'build_hybrid_eos' by calculating P on the fly.
    """
    # Unpack parameters
    log_p1, gamma_h, log_p_trans, de_ratio, c_sq = params
    
    P1 = 10**log_p1
    P_trans = 10**log_p_trans
    
    # 1. Hadronic Setup
    rho_ref = 1.85 * RHO_NUC
    K_h = P1 / (rho_ref**gamma_h)
    
    # 2. Transition Points
    rho_trans_h = (P_trans / K_h)**(1.0/gamma_h)
    eps_trans_h = rho_trans_h * c**2
    eps_trans_q = eps_trans_h + (de_ratio * RHO_NUC * c**2)
    
    current_eps = rho * c**2
    
    # 3. Calculate Candidates
    P_crust = CRUST_K * rho**CRUST_GAMMA
    P_hadron = K_h * rho**gamma_h
    P_mixed = P_trans
    P_quark = P_trans + c_sq * (current_eps - eps_trans_q)
    
    # 4. Selection Logic (The "Switch")
    # Is it Crust?
    is_crust = rho < RHO_STITCH
    # Is it Hadron? (Above crust, below transition)
    is_hadron = (rho >= RHO_STITCH) & (rho < rho_trans_h)
    # Is it Mixed? (Above hadron density, but energy < quark start)
    is_mixed = (rho >= rho_trans_h) & (current_eps < eps_trans_q)
    # Else: Quark
    
    P_val = jnp.where(is_crust, P_crust,
             jnp.where(is_hadron, P_hadron,
              jnp.where(is_mixed, P_mixed,
               P_quark)))
               
    return jnp.maximum(P_val, 0.0) # Safety floor

@jax.jit
def get_inverse_eos(P, params):
    """Calculates Density (rho) from Pressure (P)."""
    log_p1, gamma_h, log_p_trans, de_ratio, c_sq = params
    P1 = 10**log_p1
    P_trans = 10**log_p_trans
    
    rho_ref = 1.85 * RHO_NUC
    K_h = P1 / (rho_ref**gamma_h)
    rho_trans_h = (P_trans / K_h)**(1.0/gamma_h)
    eps_trans_h = rho_trans_h * c**2
    eps_trans_q = eps_trans_h + (de_ratio * RHO_NUC * c**2)
    
    # Candidates
    rho_crust = (P / CRUST_K)**(1.0 / CRUST_GAMMA)
    rho_hadron = (P / K_h)**(1.0 / gamma_h)
    # Quark: eps = (P - P_trans)/c_sq + eps_q
    rho_quark = ((P - P_trans)/c_sq + eps_trans_q) / c**2
    
    # Selection
    P_crust_limit = CRUST_K * RHO_STITCH**CRUST_GAMMA
    
    val = jnp.where(P < P_crust_limit, rho_crust,
           jnp.where(P < P_trans, rho_hadron,
            jnp.where(P == P_trans, rho_trans_h, # In jump, return start
             rho_quark)))
             
    return val

# 3. CUSTOM JAX ODE SOLVER (RK4)
# Replaces scipy.integrate.solve_ivp. 
# We use a fixed-step integrator because it compiles extremely well on JAX.

@jax.jit
def tov_equations(r, state, params):
    m, p = state
    rho = get_inverse_eos(p, params)
    
    # Standard TOV Equations
    term1 = (rho + p/c**2)
    term2 = (m + 4*jnp.pi*r**3*p/c**2)
    term3 = (1 - 2*G*m/(r*c**2))
    
    # Avoid singularity at r=0 or Event Horizon
    term3 = jnp.maximum(term3, 1e-9)
    
    dPdr = -G * term1 * term2 / (r**2 * term3)
    dMdr = 4 * jnp.pi * r**2 * rho
    
    # Stop evolution if Pressure is 0 (Surface reached)
    dPdr = jnp.where(p <= 0, 0.0, dPdr)
    dMdr = jnp.where(p <= 0, 0.0, dMdr)
    
    return jnp.array([dMdr, dPdr])

@jax.jit
def solve_one_star(central_rho, params):
    """
    Integrates one star from center to surface.
    """
    P_c = get_eos_properties(central_rho, params)
    
    # Initial Conditions (small sphere at r0)
    r0 = 10.0 # meters
    m0 = (4.0/3.0) * jnp.pi * r0**3 * central_rho
    init_state = jnp.array([m0, P_c])
    
    # Integration Grid: 3000 steps of 10 meters (30 km max radius)
    dr = 10.0
    n_steps = 3000
    
    # The Scan Loop 
    def step_fn(state, i):
        r = r0 + i * dr
        
        # RK4 Integration
        k1 = dr * tov_equations(r, state, params)
        k2 = dr * tov_equations(r + 0.5*dr, state + 0.5*k1, params)
        k3 = dr * tov_equations(r + 0.5*dr, state + 0.5*k2, params)
        k4 = dr * tov_equations(r + dr, state + k3, params)
        
        new_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        
        # Clamp pressure to 0 if it goes negative
        new_state = jnp.array([new_state[0], jnp.maximum(new_state[1], -1.0)])
        
        # Record pressure for finding surface later
        return new_state, new_state[1] 

    final_state, pressure_history = jax.lax.scan(step_fn, init_state, jnp.arange(n_steps))
    
    # Find Surface: First index where Pressure <= 0
    is_positive = pressure_history > 0
    surface_idx = jnp.sum(is_positive) # Counts how many steps P > 0
    
    # Final Mass and Radius
    # We take the state *at* the surface index
    M_final = final_state[0] # Mass accumulates, so final state is roughly correct
    R_final = r0 + surface_idx * dr
    
    return M_final / M_sun, R_final / 1000.0

# Vetorized solver

@partial(jax.jit, static_argnames=['n_points'])
def compute_curve_jax(params, n_points=50):
    # Log-space densities
    rhos = jnp.logspace(17.3, 18.6, n_points)
    
    # vmap: Map the 'solve_one_star' function over the 'rhos' array
    masses, radii = jax.vmap(solve_one_star, in_axes=(0, None))(rhos, params)
    
    return masses, radii