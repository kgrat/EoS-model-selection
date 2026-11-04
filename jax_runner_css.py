import numpy as np
import dynesty
import multiprocessing as mp
import os
import pickle
import argparse
import jax_physics_css as physics 

# Likelihood (data, KDEs, physical cuts) shared by all three runners
import eos_likelihood as like

# Logic

def prior_transform(u):
    """Maps [0,1] to Physical Parameters (CSS)."""
    log_p1 = 33.5 + 1.5 * u[0]
    gamma_h = 2.0 + 2.5 * u[1]
    log_p_trans = 33.5 + 2.0 * u[2]
    de_trans = 2.0 * u[3]
    c_sq = 0.33 + 0.67 * u[4]
    return [log_p1, gamma_h, log_p_trans, de_trans, c_sq]

def log_likelihood(params):
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
    ap.add_argument('--out', default='jax_css_results.pkl')
    args = ap.parse_args()
    print("\nStarting JAX-Accelerated Analysis")
    
    n_cpu = max(1, mp.cpu_count() - 1)
    
    with mp.Pool(processes=n_cpu) as pool:
        print(f"Running on {n_cpu} processes (each using JAX).")
        
        sampler = dynesty.NestedSampler(log_likelihood, prior_transform, ndim=5, 
                                        pool=pool, queue_size=n_cpu, 
                                        nlive=args.nlive, bound='multi')
        
        sampler.run_nested(dlogz=args.dlogz, print_progress=True)
        
    results = sampler.results
    print(f"\nLog Evidence: {results.logz[-1]:.3f}")
    
    with open(args.out, 'wb') as f:
        pickle.dump(results, f)