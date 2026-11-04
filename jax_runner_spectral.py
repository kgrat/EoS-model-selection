import numpy as np
import dynesty
import multiprocessing as mp
import os
import pickle
import argparse
import jax_physics_spectral as physics 

# Likelihood (data, KDEs, physical cuts) shared by all three runners
import eos_likelihood as like

# Priors & Likelihood for Spectral Model

def prior_transform(u):
    """
    Maps [0,1] to 4 Spectral Parameters [g0, g1, g2, g3].
    Gamma(P) = exp( sum( gamma_i * x^i ) )
    These bounds ensure Gamma stays reasonably physical (0.5 to 4.0)
    """
    # g0 (Intercept ~ Average Gamma): [0.2, 1.5] -> exp(0.2)=1.2, exp(1.5)=4.5
    g0 = 0.2 + 1.3 * u[0]
    
    # Higher order terms (Shape): Small ranges around 0
    g1 = -0.5 + 1.0 * u[1] # [-0.5, 0.5]
    g2 = -0.2 + 0.4 * u[2] # [-0.2, 0.2]
    g3 = -0.1 + 0.2 * u[3] # [-0.1, 0.1]
    
    return [g0, g1, g2, g3]

def log_likelihood(params):
    params = np.array(params)
    masses, radii = physics.compute_curve_jax(params)
    central_P = physics.central_pressures(params)
    return like.score_curve(masses, radii, central_P,
                            lambda P_max: physics.max_cs2_up_to(params, P_max))

# --- 3. RUNNER ---
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--nlive', type=int, default=1500)
    ap.add_argument('--dlogz', type=float, default=0.1)
    ap.add_argument('--out', default='jax_spectral_results.pkl')
    args = ap.parse_args()
    print("\nStarting JAX Analysis (Spectral Model)")
    
    n_cpu = max(1, mp.cpu_count() - 1)
    
    with mp.Pool(processes=n_cpu) as pool:
        print(f"Running on {n_cpu} processes.")
        
        # Spectral has 4 Parameters
        sampler = dynesty.NestedSampler(log_likelihood, prior_transform, ndim=4, 
                                        pool=pool, queue_size=n_cpu, 
                                        nlive=args.nlive, bound='multi')
        
        sampler.run_nested(dlogz=args.dlogz, print_progress=True)
        
    results = sampler.results
    print(f"\nSpectral Model Log Evidence (lnZ): {results.logz[-1]:.3f}")
    
    with open(args.out, 'wb') as f:
        pickle.dump(results, f)