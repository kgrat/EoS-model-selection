import numpy as np
import dynesty
import multiprocessing as mp
import os
import pickle
import argparse

# Import the JAX engine for the Standard Polytrope model. This file contains the JIT-compiled functions to compute M-R curves efficiently.
# We import the Standard Polytrope Physics engine here
import jax_physics_poly as physics 

# Likelihood (data, KDEs, physical cuts) shared by all three runners
import eos_likelihood as like

# Priors & Likelihood for Standard Polytrope Model

def prior_transform(u):
    """
    Maps [0,1] to 3 Physical Parameters (Piecewise Polytrope).
    Params: [logP1, logP2, logP3]
    """
    # 1. log_P1: [33.5, 35.0]
    log_p1 = 33.0 + 1.5 * u[0]
    
    # 2. log_P2: [34.0, 35.5]
    log_p2 = 34.0 + 1.5 * u[1]
    
    # 3. log_P3: [35.0, 36.5]
    log_p3 = 35.0 + 1.5 * u[2]
    
    return [log_p1, log_p2, log_p3]

def log_likelihood(params):
    # Pressure must increase between knots (a Gamma < 0 segment is thermodynamically unstable)
    if not (params[0] < params[1] < params[2]):
        return -1e10 - 1e4 * (max(params[0] - params[1], params[1] - params[2]) + 0.01)**2
    params = np.array(params)
    masses, radii = physics.compute_curve_jax(params)
    central_P = physics.central_pressures(params)
    return like.score_curve(masses, radii, central_P,
                            lambda P_max: physics.max_cs2_up_to(params, P_max))

# Runner
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--nlive', type=int, default=1500)
    ap.add_argument('--dlogz', type=float, default=0.1)
    ap.add_argument('--out', default='jax_poly_results.pkl')
    args = ap.parse_args()
    print("\n--- Starting JAX Analysis (Standard Model) ---")
    
    n_cpu = max(1, mp.cpu_count() - 1)
    
    with mp.Pool(processes=n_cpu) as pool:
        print(f"Running on {n_cpu} processes.")
        
        # ndim=3 for Polytrope
        sampler = dynesty.NestedSampler(log_likelihood, prior_transform, ndim=3, 
                                        pool=pool, queue_size=n_cpu, 
                                        nlive=args.nlive, bound='multi')
        
        sampler.run_nested(dlogz=args.dlogz, print_progress=True)
        
    results = sampler.results
    print(f"\nStandard Model Log Evidence (lnZ): {results.logz[-1]:.3f}")
    
    with open(args.out, 'wb') as f:
        pickle.dump(results, f)


