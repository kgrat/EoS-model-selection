import numpy as np
import dynesty
from scipy.stats import gaussian_kde
import multiprocessing as mp
import os
import pickle
import jax_physics_spectral as physics 

# Loading data
print("--- Loading Observational Data for KDE ---")
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
    # 1. Run JAX Physics
    masses, radii = physics.compute_curve_jax(np.array(params))
    
    masses = np.array(masses)
    radii = np.array(radii)
    
    # 2. Extract Observables
    if len(masses) == 0: return -1e10
    
    M_max = np.max(masses)
    if M_max < 2.01: return -1e10
    
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

# --- 3. RUNNER ---
if __name__ == "__main__":
    print("\nStarting JAX Analysis (Spectral Model)")
    
    n_cpu = max(1, mp.cpu_count() - 1)
    
    with mp.Pool(processes=n_cpu) as pool:
        print(f"Running on {n_cpu} processes.")
        
        # Spectral has 4 Parameters
        sampler = dynesty.NestedSampler(log_likelihood, prior_transform, ndim=4, 
                                        pool=pool, queue_size=n_cpu, 
                                        nlive=300, bound='multi')
        
        sampler.run_nested(dlogz=0.5, print_progress=True)
        
    results = sampler.results
    print(f"\nSpectral Model Log Evidence (lnZ): {results.logz[-1]:.3f}")
    
    with open('jax_spectral_results.pkl', 'wb') as f:
        pickle.dump(results, f)