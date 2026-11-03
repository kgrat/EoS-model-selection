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

@jax.jit
def get_eos_properties(rho, params):
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
    
    # Select Phase based on Density
    val = jnp.where(rho < RHO_STITCH, CRUST_K * rho**CRUST_GAMMA,
           jnp.where(rho < rho_1, k1 * rho**g1,
            jnp.where(rho < rho_2, k2 * rho**g2,
             k3 * rho**g3))) # Simple high-density extension
              
    return val

@jax.jit
def get_inverse_eos(P, params):
    log_p1, log_p2, log_p3 = params
    P1, P2, P3 = 10**log_p1, 10**log_p2, 10**log_p3
    
    P_stitch = CRUST_K * RHO_STITCH**CRUST_GAMMA
    rho_1 = 1.85 * RHO_NUC
    rho_2 = 3.70 * RHO_NUC
    rho_3 = 7.40 * RHO_NUC
    
    g1 = (jnp.log(P1) - jnp.log(P_stitch)) / (jnp.log(rho_1) - jnp.log(RHO_STITCH))
    g2 = (jnp.log(P2) - jnp.log(P1)) / (jnp.log(rho_2) - jnp.log(rho_1))
    g3 = (jnp.log(P3) - jnp.log(P2)) / (jnp.log(rho_3) - jnp.log(rho_2))
    
    k1 = P1 / rho_1**g1
    k2 = P2 / rho_2**g2
    k3 = P3 / rho_3**g3
    
    val = jnp.where(P < P_stitch, (P/CRUST_K)**(1.0/CRUST_GAMMA),
           jnp.where(P < P1, (P/k1)**(1.0/g1),
            jnp.where(P < P2, (P/k2)**(1.0/g2),
             (P/k3)**(1.0/g3))))
             
    return val

# jax tov solver (RK4)
@jax.jit
def tov_equations(r, state, params):
    m, p = state
    rho = get_inverse_eos(p, params)
    
    term1 = (rho + p/c**2)
    term2 = (m + 4*jnp.pi*r**3*p/c**2)
    term3 = jnp.maximum(1 - 2*G*m/(r*c**2), 1e-9)
    
    dPdr = -G * term1 * term2 / (r**2 * term3)
    dMdr = 4 * jnp.pi * r**2 * rho
    
    dPdr = jnp.where(p <= 0, 0.0, dPdr)
    dMdr = jnp.where(p <= 0, 0.0, dMdr)
    
    return jnp.array([dMdr, dPdr])

@jax.jit
def solve_one_star(central_rho, params):
    P_c = get_eos_properties(central_rho, params)
    
    r0 = 10.0
    m0 = (4.0/3.0) * jnp.pi * r0**3 * central_rho
    init_state = jnp.array([m0, P_c])
    
    dr = 10.0
    n_steps = 3000
    
    def step_fn(state, i):
        r = r0 + i * dr
        k1 = dr * tov_equations(r, state, params)
        k2 = dr * tov_equations(r + 0.5*dr, state + 0.5*k1, params)
        k3 = dr * tov_equations(r + 0.5*dr, state + 0.5*k2, params)
        k4 = dr * tov_equations(r + dr, state + k3, params)
        
        new_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        new_state = jnp.array([new_state[0], jnp.maximum(new_state[1], -1.0)])
        return new_state, new_state[1] 

    final_state, pressure_history = jax.lax.scan(step_fn, init_state, jnp.arange(n_steps))
    
    surface_idx = jnp.sum(pressure_history > 0)
    
    return final_state[0]/M_sun, (r0 + surface_idx*dr)/1000.0

# Vectorization 
@partial(jax.jit, static_argnames=['n_points'])
def compute_curve_jax(params, n_points=50):
    rhos = jnp.logspace(17.3, 18.6, n_points)
    masses, radii = jax.vmap(solve_one_star, in_axes=(0, None))(rhos, params)
    return masses, radii