# Mock Observing-Condition Model

This document describes the stochastic model used to generate mock
maps of observing conditions (PSF seeing, pixel noise, and Galactic
extinction) on a tiled survey footprint.

## 1 Tile geometry

The survey footprint is partitioned into a regular grid of
$N_\alpha \times N_\delta$ tiles of angular size
$\Delta\alpha \times \Delta\delta$ (default $1° \times 1°$).
Each tile $t$ has a centre $\boldsymbol{c}_t = (\alpha_t, \delta_t)$.

## 2 Tile-level draws with spatial correlation

Both the seeing and noise models decompose the tile-level value into a
spatially correlated component and an independent (uncorrelated) component.

### 2.1 Angular separation

Let $d_{ij}$ denote the flat-sky angular separation between tile
centres $i$ and $j$:

$$
d_{ij}^{2}
= \bigl[(\alpha_i - \alpha_j)\cos\bar\delta_{ij}\bigr]^{2}
+ (\delta_i - \delta_j)^{2},
\qquad
\bar\delta_{ij} = \tfrac{1}{2}(\delta_i + \delta_j).
$$

### 2.2 Correlated component

The covariance matrix for the correlated part uses a squared-exponential
(RBF) kernel:

$$
C_{ij} = \sigma_\mathrm{corr}^{2}\,
\exp\!\Bigl(-\frac{d_{ij}^{2}}{2\,\ell_\mathrm{corr}^{2}}\Bigr).
$$

The correlated tile values are drawn as

$$
\boldsymbol{s}^\mathrm{(corr)} \sim
\mathcal{N}\!\bigl(\mu_0\,\mathbf{1},\; \mathbf{C}\bigr).
$$

### 2.3 Uncorrelated component

An independent per-tile offset is added:

$$
\eta_t \sim \mathcal{N}(0,\,\sigma_\mathrm{uncorr}^{2}),
$$

drawn i.i.d. for each tile.

### 2.4 Combined tile value

$$
s_t = s_t^\mathrm{(corr)} + \eta_t.
$$

The marginal distribution of any single tile is therefore

$$
s_t \sim \mathcal{N}\!\bigl(\mu_0,\;
\sigma_\mathrm{corr}^{2} + \sigma_\mathrm{uncorr}^{2}\bigr),
$$

and the covariance between two tiles $i \neq j$ is

$$
\mathrm{Cov}(s_i, s_j)
= \sigma_\mathrm{corr}^{2}\,
\exp\!\Bigl(-\frac{d_{ij}^{2}}{2\,\ell_\mathrm{corr}^{2}}\Bigr).
$$

The ratio $\sigma_\mathrm{uncorr}/\sigma_\mathrm{corr}$ controls the
visual balance between tile-level granularity and large-scale coherence.

## 3 PSF seeing model

The seeing at pixel $p$ belonging to tile $t$ is

$$
\mathrm{PSF}(p)
= s_t\,\bigl(2 - G(r_{p,t})\bigr)
+ \epsilon_p,
$$

where $s_t \equiv s_t^{(\mathrm{PSF})}$ is the combined tile value
(Section 2.4) and

$$
G(r) = \exp\!\Bigl(-\frac{r^{2}}{2\,\sigma_\mathrm{intra}^{2}}\Bigr).
$$

At the tile centre $G=1$, so $\mathrm{PSF} = s_t + \epsilon_p$.
At the tile edges $G < 1$, so the seeing degrades as
$\mathrm{PSF} = s_t\,(2-G) > s_t$, with the degradation proportional
to the tile seeing itself.  The parameter $\sigma_\mathrm{intra}$
alone controls the steepness: a larger value yields a flatter profile
(less edge degradation).

For a $1°$ tile with $\sigma_\mathrm{intra} = 1.0°$:

| Location | $r$ | $G$ | $\mathrm{PSF}/s_t$ |
|----------|-----|-----|---------------------|
| Centre   | 0   | 1.00 | 1.00 |
| Edge     | 0.5° | 0.88 | 1.12 |
| Corner   | 0.7° | 0.78 | 1.22 |

The local distance $r_{p,t}$ from pixel $p$ to the centre of tile $t$ is

$$
r_{p,t}^{2}
= \bigl[(\alpha_p - \alpha_t)\cos\delta_t\bigr]^{2}
+ (\delta_p - \delta_t)^{2}.
$$

The pixel-level noise is i.i.d.:
$\epsilon_p \sim \mathcal{N}(0,\,\sigma_\mathrm{pix}^{2})$.

| Symbol | Description | Default |
|--------|-------------|---------|
| $\mu_0^{(\mathrm{PSF})}$ | Global mean seeing | 0.7 arcsec |
| $\sigma_\mathrm{corr}^{(\mathrm{PSF})}$ | Correlated tile scatter | 0.03 arcsec |
| $\sigma_\mathrm{uncorr}^{(\mathrm{PSF})}$ | Uncorrelated tile scatter | 0.09 arcsec |
| $\ell_\mathrm{corr}^{(\mathrm{PSF})}$ | Correlation length | 6.0 deg |
| $\sigma_\mathrm{intra}$ | Intra-tile Gaussian width | 1.0 deg |
| $\sigma_\mathrm{pix}^{(\mathrm{PSF})}$ | Pixel-level noise | 0.02 arcsec |

**Free parameters: 6.**

## 4 Pixel-noise (RMS) model

The pixel RMS at pixel $p$ in tile $t$ is

$$
\mathrm{RMS}(p) = s_t^{(\mathrm{RMS})} + \epsilon_p,
$$

where $s_t^{(\mathrm{RMS})}$ is the combined tile value (Section 2.4)
and $\epsilon_p \sim \mathcal{N}(0,\,\sigma_\mathrm{pix}^{2})$.

No intra-tile spatial structure is included: the noise level is set by
the exposure time, which is uniform within a single pointing.

| Symbol | Description | Default |
|--------|-------------|---------|
| $\mu_0^{(\mathrm{RMS})}$ | Global mean pixel RMS | 6.0 |
| $\sigma_\mathrm{corr}^{(\mathrm{RMS})}$ | Correlated tile scatter | 0.7 |
| $\sigma_\mathrm{uncorr}^{(\mathrm{RMS})}$ | Uncorrelated tile scatter | 0.6 |
| $\ell_\mathrm{corr}^{(\mathrm{RMS})}$ | Correlation length | 6.0 deg |
| $\sigma_\mathrm{pix}^{(\mathrm{RMS})}$ | Pixel-level jitter | 0.1 |

**Free parameters: 5.**

## 5 Galactic extinction

Galactic extinction is deterministic, queried from the
Schlegel, Finkbeiner & Davis (1998) dust map:

$$
A_r = R_r \times E(B-V), \qquad R_r = 2.285.
$$

**Free parameters: 0.**

## 6 Parameter summary

| # | Parameter | PSF | Noise |
|---|-----------|-----|-------|
| 1 | $\mu_0$ | 0.7 | 6.0 |
| 2 | $\sigma_\mathrm{corr}$ | 0.03 | 0.7 |
| 3 | $\sigma_\mathrm{uncorr}$ | 0.09 | 0.6 |
| 4 | $\ell_\mathrm{corr}$ | 6.0 | 6.0 |
| 5 | $\sigma_\mathrm{pix}$ | 0.02 | 0.1 |
| 6 | $\sigma_\mathrm{intra}$ | 1.0 | — |

**Total free parameters: 11** (6 PSF + 5 noise).

## 7 Implementation

| Component | Source file | Class / function |
|-----------|-----------|-----------------|
| Tile grid | `src/systematics.py` | `Tiles` |
| Correlated draws | `src/systematics.py` | `SystematicBase._correlated_tile_draws` |
| PSF model | `src/systematics.py` | `SeeingSystematic` |
| Noise model | `src/systematics.py` | `NoiseSystematic` |
| Extinction | `src/systematics.py` | `GalacticSystematic` |
| Default values | `src/config.py` | `SYSTEMATICS_CONFIG` |
