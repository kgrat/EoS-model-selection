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
    
    # 1. corner plot
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

    # Resample for physical plots
    weights = np.exp(results.logwt - lnz)
    samples = results.samples
    indices = dynesty.utils.resample_equal(np.arange(len(samples)), weights)
    subset_indices = np.random.choice(indices, 500) # Reduced to 500 for speed
    
    # 2. credible band (Mass-Radius)
    print(f"  -> Generating M-R Credible Band...")
    plt.figure(figsize=(9, 7))
    
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

    # 3. credible band (EoS: Pressure vs Density)
    print(f"  -> Generating EoS Credible Band...")
    plt.figure(figsize=(9, 7))
    rho_grid_SI = jnp.logspace(16.5, 18.6, 200) 
    
    for i, idx in enumerate(subset_indices):
        params = samples[idx]
        pressure_SI = model["phys"].get_eos_properties(rho_grid_SI, params)
        
        rho_CGS = np.array(rho_grid_SI) / 1000.0
        pressure_CGS = np.array(pressure_SI) * 10.0
        
        lbl = "Posterior EoS" if i == 0 else None
        plt.plot(rho_CGS, pressure_CGS, color=model["color"], alpha=0.03, label=lbl)

    rho_nuc_cgs = 2.7e14
    plt.axvline(rho_nuc_cgs, color='black', linestyle='-.', alpha=0.6, label=r'$\rho_{nuc}$ (Saturation)')
    
    # ONLY draw knots for the Polytrope model!
    if model["name"] == "Standard Polytrope":
        plt.axvline(1.85 * rho_nuc_cgs, color='black', linestyle=':', alpha=0.6, label=r'$1.85\rho_{nuc}$ (Knot 1)')
        plt.axvline(3.70 * rho_nuc_cgs, color='black', linestyle=':', alpha=0.4, label=r'$3.70\rho_{nuc}$ (Knot 2)')
        plt.axvline(7.40 * rho_nuc_cgs, color='black', linestyle=':', alpha=0.2, label=r'$7.40\rho_{nuc}$ (Knot 3)')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Rest Mass Density $\rho$ (g/cm$^3$)', fontsize=14, fontweight='bold')
    plt.ylabel(r'Pressure $P$ (dyn/cm$^2$)', fontsize=14, fontweight='bold')
    plt.title(f"{model['name']} Equation of State\n$\ln \mathcal{{Z}} = {lnz:.2f}$", fontsize=16)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    plt.savefig(f"{model['prefix']}_eos.png", dpi=300, bbox_inches='tight')
    plt.close()

# loop over models
for model in MODELS:
    generate_plots(model)

print("\n[DONE] All individual plots generated.")