import pickle
import numpy as np
import matplotlib.pyplot as plt
import dynesty
from dynesty import plotting as dyplot
import jax.numpy as jnp
import os

# Imports
try:
    import jax_physics_poly as poly_phys
    import jax_physics_css as css_phys
    import jax_physics_spectral as spec_phys
except ImportError:
    print("Error: Physics files not found.")
    exit()

# configuration
MODELS = [
    {
        "name": "Standard Polytrope",
        "file": "jax_poly_results.pkl",
        "phys": poly_phys,
        "color": "dodgerblue",
        "params": [r"$\log P_1$", r"$\Gamma_1$", r"$\Gamma_2$"], 
        "prefix": "poly"
    },
    {
        "name": "Hybrid (CSS)",
        "file": "jax_css_results.pkl",
        "phys": css_phys,
        "color": "crimson",
        "params": [r"$\log P_1$", r"$\Gamma_h$", r"$\log P_{trans}$", r"$\Delta\epsilon$", r"$c^2_{QM}$"],
        "prefix": "css"
    },
    {
        "name": "Spectral",
        "file": "jax_spectral_results.pkl",
        "phys": spec_phys,
        "color": "purple",
        "params": [r"$\gamma_0$", r"$\gamma_1$", r"$\gamma_2$", r"$\gamma_3$"],
        "prefix": "spectral"
    }
]

def plot_constraints():
    """Helper to draw LIGO/NICER boxes consistently."""
    plt.errorbar(11.9, 1.4, xerr=1.4, fmt='s', color='forestgreen', 
                 label='LIGO GW170817', markeredgecolor='black', capsize=5, zorder=10)
    plt.errorbar(12.71, 1.44, xerr=1.19, fmt='o', color='red', 
                 label='NICER J0030', markeredgecolor='black', capsize=5, zorder=10)
    plt.errorbar(12.39, 2.08, xerr=0.98, fmt='^', color='darkorange', 
                 label='NICER J0740', markeredgecolor='black', capsize=5, zorder=10)
    plt.axhspan(2.01, 2.15, color='black', alpha=0.15, label='J0740 Mass Limit')

def generate_plots(model):
    filename = model["file"]
    if not os.path.exists(filename):
        print(f"Skipping {model['name']} (File missing)")
        return

    print(f"Processing {model['name']}...")
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    
    lnz = results.logz[-1]
    
    # corner plot
    print(f"  -> Generating Corner Plot...")
    try:
        fig, axes = dyplot.cornerplot(results, labels=model["params"], 
                                      color=model["color"], show_titles=True, 
                                      title_fmt=".2f", quantiles=[0.16, 0.5, 0.84])
        fig.suptitle(f"{model['name']} Posteriors\n$\ln \mathcal{{Z}} = {lnz:.2f}$", fontsize=16)
        plt.savefig(f"{model['prefix']}_corner.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"     Failed corner plot: {e}")

    # credible band
    print(f"  -> Generating Credible Band...")
    plt.figure(figsize=(9, 7))
    
    # Resample
    weights = np.exp(results.logwt - lnz)
    samples = results.samples
    indices = dynesty.utils.resample_equal(np.arange(len(samples)), weights)
    subset_indices = np.random.choice(indices, 1000) 
    
    for i, idx in enumerate(subset_indices):
        params = samples[idx]
        masses, radii = model["phys"].compute_curve_jax(params)
        
        m = np.array(masses)
        r = np.array(radii)
        
        lbl = "Posterior Samples" if i == 0 else None
        plt.plot(r, m, color=model["color"], alpha=0.05, label=lbl)

    plot_constraints()
    
    plt.xlabel('Radius (km)', fontsize=14, fontweight='bold')
    plt.ylabel('Mass (Solar Masses)', fontsize=14, fontweight='bold')
    plt.title(f"{model['name']} Constraints\n$\ln \mathcal{{Z}} = {lnz:.2f}$", fontsize=16)
    plt.legend(loc='upper left', fontsize=10)
    plt.ylim(0.5, 2.5)
    plt.xlim(9, 16)
    plt.grid(True, linestyle=':', alpha=0.5)
    
    plt.savefig(f"{model['prefix']}_band.png", dpi=300, bbox_inches='tight')
    plt.close()

# loop over models
for model in MODELS:
    generate_plots(model)

print("\n[DONE] All plots generated.")