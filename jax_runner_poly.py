import numpy as np
import dynesty
from scipy.stats import gaussian_kde
import multiprocessing as mp
import os
import pickle

# Import the JAX engine for the Standard Polytrope model. This file contains the JIT-compiled functions to compute M-R curves efficiently.
# We import the Standard Polytrope Physics engine here
import jax_physics_poly as physics 

# Loading data
print("Loading Observational Data for KDE")
script_dir = os.path.dirname(os.path.realpath(__file__))

def load_kde_data(filename, col_idx, min_val, max_val):
    try:
        path = os.path.join(script_dir, filename)
        skip = 1 if 'EoS' in filename else 2
        data = np.loadtxt(path, skiprows=skip)
        
        if 'EoS' in filename: 
            mask = (data[:,0] > min_val) & (data[:,0] < max_val)
            samples = data[mask, 4] 
        else:
            mask = (data[:,1] > min_val) & (data[:,1] < max_val)
            samples = data[mask, 0] 
            
        return gaussian_kde(samples)
    except Exception as e:
        print(f"[FAIL] {filename}: {e}")
        return None

kde_ligo = load_kde_data('EoS-insensitive_posterior_samples.dat', 0, 1.35, 1.45)
kde_j0030 = load_kde_data('J0030_data.dat', 1, 1.35, 1.45)
kde_j0740 = load_kde_data('J0740_data.dat', 1, 2.0, 2.15)

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
    # Run JAX Physics (Standard Polytrope)
    # The JAX engine solves for the M-R curve instantly
    masses, radii = physics.compute_curve_jax(np.array(params))
    
    masses = np.array(masses)
    radii = np.array(radii)
    
    # 2. Extract Observables
    if len(masses) == 0: return -1e10
    
    M_max = np.max(masses)
    if M_max < 2.01: return -1e10 # Must support J0740
    
    # Sort & Interpolate
    idxs = np.argsort(masses)
    m_s = masses[idxs]
    r_s = radii[idxs]
    m_u, unique_indices = np.unique(m_s, return_index=True)
    r_u = r_s[unique_indices]
    
    if len(m_u) < 5: return -1e10
    
    try:
        r_14 = np.interp(1.4, m_u, r_u)
        r_20 = np.interp(2.08, m_u, r_u) if M_max > 2.08 else 0
    except:
        return -1e10

    if r_14 <= 0: return -1e10

    # 3. KDE Scoring
    score = 0
    if kde_ligo: score += kde_ligo.logpdf(r_14)[0]
    if kde_j0030: score += kde_j0030.logpdf(r_14)[0]
    if kde_j0740 and r_20 > 0: score += kde_j0740.logpdf(r_20)[0]
    
    return score

# Runner
if __name__ == "__main__":
    print("\n--- Starting JAX Analysis (Standard Model) ---")
    
    n_cpu = max(1, mp.cpu_count() - 1)
    
    with mp.Pool(processes=n_cpu) as pool:
        print(f"Running on {n_cpu} processes.")
        
        # ndim=3 for Polytrope
        sampler = dynesty.NestedSampler(log_likelihood, prior_transform, ndim=3, 
                                        pool=pool, queue_size=n_cpu, 
                                        nlive=300, bound='multi')
        
        sampler.run_nested(dlogz=0.5, print_progress=True)
        
    results = sampler.results
    print(f"\nStandard Model Log Evidence (lnZ): {results.logz[-1]:.3f}")
    
    with open('jax_poly_results.pkl', 'wb') as f:
        pickle.dump(results, f)


