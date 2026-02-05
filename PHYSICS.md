# ⚛️ Physics & Models Documentation

This document details the theoretical frameworks and equations implemented in the JAX solver.

## 1. The Tolman-Oppenheimer-Volkoff (TOV) Equations

The structure of a static, spherically symmetric relativistic star is governed by the TOV equations. We solve these for pressure $P(r)$ and mass $m(r)$:

$$
\frac{dP}{dr} = -\frac{G(\epsilon + P/c^2)(m + 4\pi r^3 P/c^2)}{r(r - 2Gm/c^2)}
$$

$$
\frac{dm}{dr} = 4\pi r^2 \rho
$$

Where $\epsilon$ is the energy density ($\epsilon \approx \rho c^2$). The solver integrates from the center ($P_c$) to the surface ($P=0$).

## 2. The Equation of State (EoS) Models

We test three distinct parameterizations connecting Pressure ($P$) and Density ($\rho$).

### A. Piecewise Polytrope (The "Standard" Model)

*Based on Read et al. (2009).*
This model assumes the EoS is composed of discrete power-law segments stitched together at fixed densities ($\rho_1, \rho_2, \rho_3$).

$$
P(\rho) = K_i \rho^{\Gamma_i}
$$

* **Parameters:** $\{\log P_1, \Gamma_1, \Gamma_2\}$ (3 parameters).

* **Behavior:** Allows for sharp changes in stiffness ("soft" to "stiff") at specific densities.

* **Result:** Fits the data best because it captures the specific "Soft-then-Stiff" requirement of LIGO/NICER.

### B. Constant Speed of Sound (CSS) / Hybrid Model

*Based on Alford, Han, & Prakash (2013).*
This model tests for a **First-Order Phase Transition** (e.g., Hadron $\to$ Quark). It introduces a discontinuity in the energy density.

* **Hadronic Phase:** Standard Polytrope.

* **Transition:** At $P_{trans}$, density jumps from $\epsilon_h \to \epsilon_q$ (Latent Heat $\Delta \epsilon$).

* **Quark Phase:** The speed of sound $c_{QM}^2$ is constant.
  

  $$
  P(\epsilon) = P_{trans} + c_{QM}^2 (\epsilon - \epsilon_{q})
  $$

* **Parameters:** $\{\log P_1, \Gamma, P_{trans}, \Delta \epsilon, c_{QM}^2\}$ (5 parameters).

* **Result:** Fits well, but the extra complexity is penalized by the Bayesian Evidence.

### C. Spectral Parameterization

*Based on Lindblom (2010).*
This model enforces smoothness by expanding the adiabatic index $\Gamma$ as a polynomial in log-pressure space.

$$
\Gamma(x) = \exp\left( \sum_{k=0}^3 \gamma_k x^k \right)
$$

* **Parameters:** $\{\gamma_0, \gamma_1, \gamma_2, \gamma_3\}$ (4 parameters).

* **Result:** **Disfavored.** The requirement for smoothness causes the model to oscillate unphysically when trying to fit the sharp constraints, failing to support high-mass stars while maintaining a small radius.

## 3. Observational Constraints (Likelihoods)

The models are vetted against three key datasets using Kernel Density Estimation (KDE):

1. **LIGO GW170817:** Constrains the **Tidal Deformability (**$\Lambda$**)**.

   * *Physics:* A "soft" EoS (small radius) produces less tidal distortion.

   * *Constraint:* Favors $R_{1.4} < 13.5$ km.

2. **NICER PSR J0030+0451:** A standard mass pulsar (\~1.4 $M_{\odot}$).

   * *Constraint:* Provides a broad radius measurement.

3. **NICER PSR J0740+6620:** The most massive known neutron star (\~2.08 $M_{\odot}$).

   * *Physics:* Requires a very "stiff" EoS core to prevent collapse into a black hole.

   * *Constraint:* Sets a hard lower limit on stiffness at high densities.

## 4. Bayesian Statistical Framework
We employ **Bayesian Model Selection** to compare the EoS parameterizations. The probability of a model $M$ given data $D$ is proportional to the **Bayesian Evidence** ($\mathcal{Z}$):

$$
P(M|D) \propto P(D|M) P(M) = \mathcal{Z} \cdot P(M)
$$

Where the Evidence $\mathcal{Z}$ is the integral of the likelihood $\mathcal{L}$ over the entire prior volume $\pi(\theta)$:

$$
\mathcal{Z} = \int \mathcal{L}(D|\theta, M) \pi(\theta|M) d\theta
$$

To compare two models (e.g., Hybrid vs. Standard), we calculate the **Bayes Factor** ($\mathcal{B}_{AB}$):

$$
\mathcal{B}_{AB} = \frac{\mathcal{Z}_A}{\mathcal{Z}_B} = \exp(\ln \mathcal{Z}_A - \ln \mathcal{Z}_B)
$$

**Interpretation (Jeffreys Scale):**
* $\ln \mathcal{B} > 1.0$: Substantial Evidence
* $\ln \mathcal{B} > 2.5$: Strong Evidence
* $\ln \mathcal{B} > 5.0$: Decisive Evidence

## 5. Prior Distributions

We utilize **Uniform Priors** for all parameters to remain agnostic within physically motivated bounds.

| Model | Parameter | Range | Description |
| :--- | :--- | :--- | :--- |
| **Standard** | $\log P_1$ | $[33.0, 35.0]$ | Pressure at $1.85\rho_{nuc}$ |
| | $\Gamma_1$ | $[2.0, 4.5]$ | Stiffness index 1 |
| | $\Gamma_2$ | $[1.1, 4.5]$ | Stiffness index 2 |
| **Hybrid (CSS)** | $\log P_{trans}$ | $[33.5, 35.5]$ | Transition Pressure (Pa) |
| | $\Delta \epsilon / \rho_{nuc}$ | $[0.0, 2.0]$ | Energy Density Jump |
| | $c_{QM}^2$ | $[0.33, 1.0]$ | Quark Speed of Sound ($c=1$) |
| **Spectral** | $\gamma_0$ | $[0.2, 1.5]$ | Intercept ($\approx$ Avg Gamma) |
| | $\gamma_{1,2,3}$ | $[-0.5, 0.5]$ | Shape coefficients |

## 6. Likelihood Function ($\ln \mathcal{L}$)

Unlike standard analyses that assume Gaussian errors for Mass/Radius, we account for the non-Gaussian correlations in the observational data using **Kernel Density Estimation (KDE)**.

The total log-likelihood is the sum of the log-probabilities from the independent datasets:

$$
\ln \mathcal{L}(\theta) = \ln P_{LIGO}(R_{1.4}(\theta)) + \ln P_{J0030}(R_{1.4}(\theta)) + \ln P_{J0740}(R_{2.08}(\theta))
$$

* **Physics Cuts:** The likelihood is set to $-\infty$ (0 probability) if:
  1. The Maximum Mass $M_{max} < 2.01 M_{\odot}$ (Violates J0740 existence).
  2. The EoS becomes acausal ($c_s > c$).
  3. The star collapses before reaching the observed pulsar masses.