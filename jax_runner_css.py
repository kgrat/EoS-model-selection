import numpy as np
import dynesty
from scipy.stats import gaussian_kde
import multiprocessing as mp
import os
import pickle
import jax_physics_css as physics 

# Loading data
print("--- Loading Observational Data ---")
script_dir = os.path.dirname(os.path.realpath(__file__))

# Helper to load safely
def load_kde_data(filename, col_idx, min_val, max_val):
    try:
        path = os.path.join(script_dir, filename)
        # Skip different rows for different files logic
        skip = 1 if 'EoS' in filename else 2
        data = np.loadtxt(path, skiprows=skip)
        
        # Logic for LIGO vs NICER file structures
        if 'EoS' in filename: 
            # LIGO: Check Mass1 (col 0)
            mask = (data[:,0] > min_val) & (data[:,0] < max_val)
            samples = data[mask, 4] # Radius1
        else:
            # NICER: Check Mass (Col 1)
            mask = (data[:,1] > min_val) & (data[:,1] < max_val)
            samples = data[mask, 0] # Radius
            
        kde = gaussian_kde(samples)
        print(f"[OK] {filename} KDE built ({len(samples)} samples).")
        return kde
    except Exception as e:
        print(f"[FAIL] {filename}: {e}")
        return None

kde_ligo = load_kde_data('EoS-insensitive_posterior_samples.dat', 0, 1.35, 1.45)
kde_j0030 = load_kde_data('J0030_data.dat', 1, 1.35, 1.45)
kde_j0740 = load_kde_data('J0740_data.dat', 1, 2.0, 2.15)

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
    # Note: We cast params to numpy array for dynesty, but JAX handles it
    # JIT-compiled function. First run allows JAX to compile.
    
    # We ask JAX for 50 points on the curve instantly
    masses, radii = physics.compute_curve_jax(np.array(params))
    
    # Convert JAX arrays back to Numpy for SciPy KDE (fast, negligible cost)
    masses = np.array(masses)
    radii = np.array(radii)
    
    # 2. Extract Observables (Interpolation)
    if len(masses) == 0: return -1e10
    
    M_max = np.max(masses)
    if M_max < 2.01: return -1e10 # J0740 Check
    
    # Sort for interpolation
    idxs = np.argsort(masses)
    m_s = masses[idxs]
    r_s = radii[idxs]
    
    # Remove duplicates
    m_u, unique_indices = np.unique(m_s, return_index=True)
    r_u = r_s[unique_indices]
    
    if len(m_u) < 5: return -1e10
    
    # Interpolate
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
    print("\nStarting JAX-Accelerated Analysis")
    
    n_cpu = max(1, mp.cpu_count() - 1)
    
    with mp.Pool(processes=n_cpu) as pool:
        print(f"Running on {n_cpu} processes (each using JAX).")
        
        sampler = dynesty.NestedSampler(log_likelihood, prior_transform, ndim=5, 
                                        pool=pool, queue_size=n_cpu, 
                                        nlive=300, bound='multi')
        
        sampler.run_nested(dlogz=0.5, print_progress=True)
        
    results = sampler.results
    print(f"\nLog Evidence: {results.logz[-1]:.3f}")
    
    with open('jax_css_results.pkl', 'wb') as f:
        pickle.dump(results, f)