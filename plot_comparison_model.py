import pickle
import numpy as np
import matplotlib.pyplot as plt
import dynesty
import jax.numpy as jnp
import os

# Imports
try:
    import jax_physics_poly as poly_phys
    import jax_physics_css as css_phys
    import jax_physics_spectral as spec_phys
except ImportError as e:
    print("CRITICAL ERROR: Physics files missing.")
    exit()

# Load Evidence
def get_logz(filename):
    if not os.path.exists(filename): return -np.inf
    with open(filename, 'rb') as f:
        res = pickle.load(f)
    return res.logz[-1]

lnz_poly = get_logz("jax_poly_results.pkl")
lnz_css = get_logz("jax_css_results.pkl")
lnz_spec = get_logz("jax_spectral_results.pkl")

scores = {"Standard": lnz_poly, "Hybrid": lnz_css, "Spectral": lnz_spec}
winner_name = max(scores, key=scores.get)
winner_score = scores[winner_name]

def make_label(name, my_score):
    if name == winner_name:
        return f"{name} (Winner, $\ln \mathcal{{Z}}={my_score:.1f}$)"
    delta = my_score - winner_score
    return f"{name} ($\Delta \ln \mathcal{{Z}}={delta:.1f}$)"

# configuration
MODELS = [
    {
        "name": "Spectral",
        "file": "jax_spectral_results.pkl",
        "phys": spec_phys,
        "color": "purple",
        "label": make_label("Spectral", lnz_spec),
        "zorder": 1,
        "alpha": 0.020,
        "n_plot": 200
    },
    {
        "name": "Hybrid",
        "file": "jax_css_results.pkl",
        "phys": css_phys,
        "color": "crimson",
        "label": make_label("Hybrid", lnz_css),
        "zorder": 2,
        "alpha": 0.025,
        "n_plot": 400
    },
    {
        "name": "Standard",
        "file": "jax_poly_results.pkl",
        "phys": poly_phys,
        "color": "dodgerblue",
        "label": make_label("Standard", lnz_poly),
        "zorder": 3,
        "alpha": 0.04,
        "n_plot": 600
    }
]

def plot_bands(model_dict, ax):
    filename = model_dict["file"]
    if not os.path.exists(filename): return

    with open(filename, 'rb') as f:
        results = pickle.load(f)
    
    weights = np.exp(results.logwt - results.logz[-1])
    samples = results.samples
    indices = dynesty.utils.resample_equal(np.arange(len(samples)), weights)
    subset_indices = np.random.choice(indices, model_dict["n_plot"])
    
    for i, idx in enumerate(subset_indices):
        params = samples[idx]
        masses, radii = model_dict["phys"].compute_curve_jax(params)
        m = np.array(masses)
        r = np.array(radii)
        
        mask = (r > 8) & (r < 17) & (m > 0.1) & (m < 3.0)
        ax.plot(r[mask], m[mask], color=model_dict["color"], 
                alpha=model_dict["alpha"], zorder=model_dict["zorder"], lw=1.0)

# Plotting
plt.figure(figsize=(10, 8))
ax = plt.gca()

for model in MODELS:
    plot_bands(model, ax)

# Dummy lines for legend
for model in MODELS:
    ax.plot([], [], color=model["color"], label=model["label"], linewidth=3)

# Constraints
plt.errorbar(11.9, 1.4, xerr=1.4, fmt='s', color='forestgreen', 
             label='LIGO GW170817', markeredgecolor='black', markersize=8, capsize=5, zorder=10, lw=2)
plt.errorbar(12.71, 1.44, xerr=1.19, fmt='o', color='red', 
             label='NICER J0030', markeredgecolor='black', markersize=8, capsize=5, zorder=10, lw=2)
plt.errorbar(12.39, 2.08, xerr=0.98, fmt='^', color='darkorange', 
             label='NICER J0740', markeredgecolor='black', markersize=8, capsize=5, zorder=10, lw=2)

plt.axhspan(2.01, 2.15, color='black', alpha=0.15, zorder=0)
plt.text(9.1, 2.03, "J0740 Mass Limit", fontsize=10, color='gray', fontweight='bold', ha='left')

# Formatting
plt.xlabel('Radius (km)', fontsize=14, fontweight='bold')
plt.ylabel('Mass (Solar Masses)', fontsize=14, fontweight='bold')
plt.title('Neutron Star EoS Selection', fontsize=16, fontweight='bold', pad=15)

# Legend
plt.legend(loc='upper right', fontsize=9, framealpha=1.0, edgecolor='black', shadow=True)

plt.ylim(0.5, 2.5)
plt.xlim(9, 16)
plt.minorticks_on()
plt.grid(True, which='major', linestyle='--', alpha=0.5)
plt.grid(True, which='minor', linestyle=':', alpha=0.2)

plt.savefig("Neutron_Star_Model.png", dpi=300, bbox_inches='tight')
print("\n[SUCCESS] Plot saved.")
plt.show()