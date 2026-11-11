# Physics and Models

This document lists the equations and parameter choices implemented in the JAX solvers. All code works in SI units: pressure $P$ in Pa, rest-mass density $\rho$ in kg/m$^3$, energy density $\epsilon$ in J/m$^3$.

## 1. The Tolman-Oppenheimer-Volkoff (TOV) equations

A static, spherically symmetric star in general relativity satisfies

$$
\frac{dm}{dr} = 4\pi r^2 \frac{\epsilon}{c^2}
$$

$$
\frac{dP}{dr} = -\frac{G\left(\epsilon/c^2 + P/c^2\right)\left(m + 4\pi r^3 P/c^2\right)}{r^2\left(1 - 2Gm/(rc^2)\right)}
$$

Here $\epsilon$ is the **total energy density** (rest mass plus internal energy), not the rest-mass density $\rho$. Gravity couples to all of the energy, so $\epsilon$ appears in both equations. Using $\rho$ instead under-counts the mass-energy at high density and overestimates the maximum mass (see Corrections in the README).

Integration starts at $r_0 = 10$ m with $m(r_0) = \tfrac{4}{3}\pi r_0^3\,\epsilon_c/c^2$ and $P(r_0) = P_c$, and uses fixed-step RK4 with $dr = 10$ m out to 30 km. The radius $R$ is where $P$ first reaches zero and the mass is $M = m(R)$. All the solver needs from the EoS is $\epsilon(P)$; each model below provides it.

## 2. Energy density of a polytrope

For a cold, barotropic EoS the first law of thermodynamics is

$$
d\!\left(\frac{\epsilon}{\rho}\right) = -P\, d\!\left(\frac{1}{\rho}\right).
$$

For a polytropic piece $P = K_i \rho^{\Gamma_i}$ on $\rho_{i-1} \le \rho < \rho_i$ this integrates to (Read et al. 2009, Eqs. 6-7)

$$
\epsilon(\rho) = (1 + a_i)\,\rho c^2 + \frac{K_i}{\Gamma_i - 1}\,\rho^{\Gamma_i},
$$

$$
a_i = \frac{\epsilon(\rho_{i-1})}{\rho_{i-1} c^2} - 1 - \frac{K_i}{(\Gamma_i - 1)\,c^2}\,\rho_{i-1}^{\Gamma_i - 1}.
$$

The constant $a_i$ makes $\epsilon$ continuous at each dividing density. For the crust, $a_0 = 0$, so $\epsilon \to \rho c^2$ as $\rho \to 0$. Since $K_i\rho^{\Gamma_i} = P$, the second term is $P/(\Gamma_i - 1)$, the internal energy. Inside the ODE, $\epsilon(P)$ is computed as $P \to \rho$ (inverse EoS) $\to \epsilon(\rho)$, using `jnp` gathers over the segment tables so it stays JIT- and `vmap`-compatible.

The sound speed follows from $c_s^2 = dP/d\epsilon = \Gamma P/(\epsilon + P)$ (in units of $c^2$).

## 3. The EoS models

### A. Piecewise polytrope

*Read et al. (2009), with a different crust and knot choice.*

* **Crust:** $P = K_c \rho^{\Gamma_c}$ with $K_c = 1.0557\times10^5$ (SI) and $\Gamma_c = 1.58425$ for $\rho < 0.5\,\rho_{nuc}$, where $\rho_{nuc} = 2.7\times10^{17}$ kg/m$^3$.
* **Core:** three knots at $\rho_1 = 1.85\,\rho_{nuc}$, $\rho_2 = 3.7\,\rho_{nuc}$ and $\rho_3 = 7.4\,\rho_{nuc}$, with free pressures $P_1, P_2, P_3$. The segment $[0.5\,\rho_{nuc}, \rho_1)$ joins the crust to $(\rho_1, P_1)$. Above $\rho_2$ the segment through $(\rho_2, P_2)$ and $(\rho_3, P_3)$ is continued to all higher densities.
* **Parameters:** $\{\log_{10}P_1, \log_{10}P_2, \log_{10}P_3\}$ ($P$ in Pa).
* **Energy density:** Section 2, with four segments.

The single-segment crust differs from the four-piece SLy crust in Read et al.; at the stitch it gives about 2.7 times the SLy pressure. Running SLy's core through this crust gives $R_{1.4}$ about 0.45 km larger than with the SLy crust (see `validation_output.txt`).

### B. Constant speed of sound (CSS) hybrid model

*Alford, Han & Prakash (2013).* Tests for a first-order hadron-quark transition.

* **Crust:** as above, for $P < P_{stitch} = K_c (0.5\rho_{nuc})^{\Gamma_c}$.
* **Hadronic phase** ($P_{stitch} \le P < P_{trans}$): $P = K_h \rho^{\Gamma_h}$ with $K_h = P_1/(1.85\rho_{nuc})^{\Gamma_h}$, and
  $\epsilon_h = (1 + a_h)\rho c^2 + P/(\Gamma_h - 1)$. $a_h$ makes $\epsilon$ continuous at $P_{stitch}$.
* **Transition** at $P_{trans}$: the energy density jumps from $\epsilon_h(P_{trans})$ to
  $\epsilon_q = \epsilon_h(P_{trans}) + \Delta\epsilon$, with pressure held constant (latent heat).
* **Quark phase** ($P \ge P_{trans}$):

  $$
  P(\epsilon) = P_{trans} + c_s^2\,(\epsilon - \epsilon_q).
  $$

* **Parameters:** $\{\log_{10}P_1, \Gamma_h, \log_{10}P_{trans}, \Delta\epsilon/(\rho_{nuc}c^2), c_s^2/c^2\}$.

Stars are labelled by their central energy density $\epsilon_c$; $P_c$ is found by bisection on $\epsilon(P)$. The rest-mass density in the quark phase (used only for plots) follows from $d\rho/\rho = d\epsilon/(\epsilon + P)$. The jump in $\rho$ at the transition comes from continuity of the baryon chemical potential $(\epsilon + P)/n$.

The crust hands over to the hadronic polytrope at fixed pressure $P_{stitch}$. In general the hadronic $\rho$ at that pressure differs from $0.5\rho_{nuc}$, so $\rho$, but not $P$ or $\epsilon$, is discontinuous there. This has no effect on the TOV solution, which only uses $\epsilon(P)$.

### C. Spectral parameterization

*Lindblom (2010).* The adiabatic index is expanded in $x = \ln(P/P_0)$, with $P_0 = P_{stitch}$:

$$
\Gamma(x) = \exp\left(\sum_{k=0}^3 \gamma_k x^k\right).
$$

The energy density follows from Lindblom's integrals (his Eqs. 8-9):

$$
\mu(x) = \exp\left(-\int_0^x \frac{dx'}{\Gamma(x')}\right), \qquad
\epsilon(x) = \frac{\epsilon_0}{\mu(x)} + \frac{1}{\mu(x)}\int_0^x \frac{\mu(x')\,P_0 e^{x'}}{\Gamma(x')}\,dx', \qquad
\rho(x) = \frac{\rho_0}{\mu(x)}.
$$

$\epsilon_0 = \rho_0 c^2 + P_0/(\Gamma_c - 1)$ is the crust energy density at the stitch. The integrals are tabulated once per parameter vector, with a cumulative trapezoid on 4000 points up to $P = 10^{37}$ Pa, and $\epsilon(P)$ is interpolated in log space. For constant $\Gamma$ the table agrees with the analytic polytrope to $3\times10^{-7}$.

* **Parameters:** $\{\gamma_0, \gamma_1, \gamma_2, \gamma_3\}$.

With $x$ reaching about 10 at the highest central pressures, the cubic term can dominate. Over much of the prior, $\Gamma$ either becomes very large (acausal) or very small ($\epsilon$ and $\rho$ overflow) at high pressure. Stars that reach the overflow region are dropped.

## 4. Observational likelihood

The likelihood (`eos_likelihood.py`) is an approximation built from one-dimensional kernel density estimates (KDEs) of **radius at a fixed mass**. It is not a tidal-deformability likelihood.

$$
\ln\mathcal{L}(\theta) = \ln \hat p_{GW}(R_{1.4}) + \ln \hat p_{J0030}(R_{1.4}) + \ln \left\langle \hat p_{J0740}\big(R(M)\big) \right\rangle_{M}
$$

* $\hat p_{GW}$: KDE of the GW170817 radius samples from the EoS-insensitive analysis (Abbott et al. 2018), using the samples with $1.35 < m_1 < 1.45\,M_\odot$. These radii were derived from the measured tidal deformabilities with EoS-insensitive (quasi-universal) relations, so $\Lambda$ enters only through that external step.
* $\hat p_{J0030}$: KDE of NICER PSR J0030+0451 radius samples (Miller et al. 2019) with $1.35 < M < 1.45\,M_\odot$.
* $\hat p_{J0740}$: KDE of NICER PSR J0740+6620 radius samples (Dittmann et al. 2024; NICER + XMM-Newton, fully ionized hydrogen atmosphere) with $2.0 < M < 2.15\,M_\odot$. It is averaged over the pulsar's mass distribution in that window, using 200 quantiles of the mass samples. Masses the EoS cannot reach contribute zero. An EoS whose $M_{max}$ only just exceeds $2.01\,M_\odot$ is therefore partly penalized, rather than having the term skipped. For EoSs that reach $2.15\,M_\odot$ the term equals $\ln\hat p_{J0740}(R_{2.08})$ to within about 0.01.

$R(M)$ is read off the **stable branch**: the stretches of the M-R curve where $M$ increases with central density. Where several stable branches contain the same mass (twin stars), the lowest-density one is used.

**Physical cuts.** The cuts below are applied as $\ln\mathcal{L} \approx -10^{10}$, with a small slope toward the allowed region:

1. $M_{max} < 2.01\,M_\odot$.
2. **Causality:** $c_s^2 = dP/d\epsilon > c^2$ anywhere below the central pressure of the maximum-mass star.
3. **Monotonic pressure** (polytrope): $P_1 < P_2 < P_3$, so every segment has $\Gamma > 0$.

These cuts remove prior volume without renormalizing the prior. The evidence therefore includes the fraction of each model's prior that is physical: a model that spends most of its prior on acausal EoSs is penalized for it.

Remaining limitations:

1. Mass-radius posteriors are reduced to radius distributions, so the mass-radius correlation within each measurement is discarded.
2. The GW170817 and J0030 terms both constrain $R_{1.4}$ and are multiplied as if independent.

## 5. Bayesian model selection

The evidence is

$$
\mathcal{Z} = \int \mathcal{L}(D|\theta, M)\,\pi(\theta|M)\,d\theta,
$$

estimated with `dynesty` static nested sampling (`bound='multi'`). The Bayes factor between models A and B is $\mathcal{B}_{AB} = \exp(\ln\mathcal{Z}_A - \ln\mathcal{Z}_B)$.

On the Jeffreys scale, $|\ln\mathcal{B}| \approx 1$-$2.5$ is "substantial" and $2.5$-$5$ "strong". Two caveats apply to this project:

* $\mathcal{Z}$ depends directly on the prior volume. Widening a uniform prior range in a direction the data do not constrain lowers $\ln\mathcal{Z}$ by the log of the widening factor without any change in fit quality. The three models have unrelated parameterizations and prior ranges, so a $\ln\mathcal{Z}$ difference of about 2 is modest and should not be read as a statement about the physics alone.
* The likelihood is the KDE approximation above, so all the evidences inherit its limitations.

## 6. Priors

All priors are uniform over the ranges below (from the `prior_transform` functions in the runners).

| Model | Parameter | Range |
| :--- | :--- | :--- |
| Polytrope | $\log_{10}P_1$ [Pa] | $[33.0, 34.5]$ |
| | $\log_{10}P_2$ [Pa] | $[34.0, 35.5]$ |
| | $\log_{10}P_3$ [Pa] | $[35.0, 36.5]$ |
| CSS | $\log_{10}P_1$ [Pa] | $[33.5, 35.0]$ |
| | $\Gamma_h$ | $[2.0, 4.5]$ |
| | $\log_{10}P_{trans}$ [Pa] | $[33.5, 35.5]$ |
| | $\Delta\epsilon/(\rho_{nuc}c^2)$ | $[0, 2]$ |
| | $c_s^2/c^2$ | $[0.33, 1.0]$ |
| Spectral | $\gamma_0$ | $[0.2, 1.5]$ |
| | $\gamma_1$ | $[-0.5, 0.5]$ |
| | $\gamma_2$ | $[-0.2, 0.2]$ |
| | $\gamma_3$ | $[-0.1, 0.1]$ |

The polytrope prior ranges overlap, so they include $P_2 \le P_1$ or $P_3 \le P_2$ (a segment with $\Gamma \le 0$). These are rejected by the likelihood (Section 4).

## 7. Validation

`validate_energy_density.py` checks the corrected solver (output in `validation_output.txt`):

* SLy (Read et al. 2009, Tables II-III, 4-piece crust plus 3-piece core): $M_{max} = 2.048\,M_\odot$ and $R_{1.4} = 11.70$ km, against the published 2.049 and 11.736 km. The old $\rho$-based TOV gives 2.456 and 12.20 km for the same EoS. An independent `scipy` DOP853 solver agrees with the JAX RK4 solver to 0.001% in $M_{max}$ and 13 m in $R_{1.4}$.
* $\epsilon$ is continuous at every segment boundary to $10^{-12}$ or better, and the CSS jump equals $\Delta\epsilon$.
* $\epsilon \ge \rho c^2$ everywhere. Causality is reported here and enforced in the likelihood.

## References

* Read, Lackey, Owen & Friedman, Phys. Rev. D 79, 124032 (2009).
* Lindblom, Phys. Rev. D 82, 103011 (2010).
* Alford, Han & Prakash, Phys. Rev. D 88, 083013 (2013).
* Abbott et al. (LIGO/Virgo), Phys. Rev. Lett. 121, 161101 (2018).
* Miller, M. C., Lamb, F. K., Dittmann, A. J., et al., ApJL 887, L24 (2019).
* Dittmann, A. J., Miller, M. C., Lamb, F. K., et al., ApJ 974, 295 (2024).
