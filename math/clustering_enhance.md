# Clustering Enhancement due to spatial variation in redshift distributions

> **Context:** Observational cosmology. Galaxy survey. Galaxy clustering. Two-point correlation of galaxy positions.

# Qualitative Discussion

> **Problem:** $ n(z) $ varies across the survey footprint, primarily due to observational selection effects (e.g. varying observing conditions). The global redshift distribution $ \bar n(z) $ differs from the local $ n(z,\theta) $. Modeling the angular two-point correlation function (2PCF) using the global $ \bar n(z) $ leads to an underestimation of the true clustering amplitude.

---

# Math

> **Goal:** Write the angular 2PCF in terms of the local $ n(z,\theta) $, then relate the difference between the true and modeled clustering amplitudes to the spatial variance of $ n(z,\theta) $.

---

### Projected galaxy overdensity

The projected galaxy overdensity at angular position $ \boldsymbol{\theta} $ is
$$
\delta_g(\boldsymbol{\theta})
=
\int dz\, n(z,\boldsymbol{\theta})\,\delta_{m,\rm 3D}(z,\boldsymbol{\theta}) b(z),
$$
with normalization
$$
\int dz\, n(z,\boldsymbol{\theta}) = 1 \qquad \forall\,\boldsymbol{\theta}.
$$

The angular two-point correlation function at separation $ \theta $ is
$$
w(\theta)
\equiv
\left\langle
\delta_g(\boldsymbol{\theta}_1)\,
\delta_g(\boldsymbol{\theta}_2)
\right\rangle_{|\boldsymbol{\theta}_1-\boldsymbol{\theta}_2|=\theta},
$$
where $ \langle\cdot\rangle $ denotes an average over all sky pairs at fixed angular separation.

Substituting the projected overdensity,
$$
\begin{aligned}
w(\theta)
&=
\int dz_1\,dz_2\,
\left\langle
n(z_1,\boldsymbol{\theta}_1)\,
n(z_2,\boldsymbol{\theta}_2)\,
\delta_m(z_1,\boldsymbol{\theta}_1)\,
\delta_m(z_2,\boldsymbol{\theta}_2)
\right\rangle .
\end{aligned}
$$

---

### Independence assumption

Assume that the redshift selection field $ n(z,\theta) $ is statistically independent of the matter density field $ \delta_m $:
$$
\langle n\,n\,\delta_m\,\delta_m\rangle
=
\langle n\,n\rangle\,
\langle \delta_m\,\delta_m\rangle .
$$

### Limber / narrow-kernel approximation

For angular clustering, correlated pairs satisfy $ z_1 \simeq z_2 $, such that
$$
\langle \delta_m(z_1)\delta_m(z_2)\rangle
\;\rightarrow\;
\xi_m(\theta;z)\,\delta_D(z_1-z_2).
$$

This yields the true angular correlation function
$$
w_{\mathrm{true}}(\theta)
=
\int dz\,
\left\langle n^2(z,\theta)\right\rangle_\theta\,
\xi_m(\theta;z).
$$

---

### Global redshift distribution and model prediction

Define the global redshift distribution
$$
\bar n(z) \equiv \left\langle n(z,\theta)\right\rangle_\theta .
$$

Using ( $\bar n(z)$ ) instead of the local $ n(z,\theta) $, the modeled correlation function is
$$
w_{\mathrm{model}}(\theta)
=
\int dz\,
\bar n^2(z)\,
\xi_m(\theta;z).
$$


Write
$$
n(z,\theta) = \bar n(z) + \delta n(z,\theta),
\qquad
\left\langle \delta n(z,\theta) \right\rangle_\theta = 0 .
$$

Then
$$
\left\langle n^2(z,\theta)\right\rangle_\theta
=
\bar n^2(z)
+
\left\langle \delta n^2(z,\theta)\right\rangle_\theta
$$

---

### Clustering enhancement due to spatial variation of $ n(z,\theta) $ 

The difference between the true and modeled clustering amplitudes is
$$
\Delta w(\theta)
=
w_{\mathrm{true}}(\theta) - w_{\mathrm{model}}(\theta)
=
\int dz\,
\left\langle \delta n^2(z,\theta)\right\rangle_\theta
\xi_m(\theta;z).
$$

Equivalently, the multiplicative bias is
$$
\frac{w_{\mathrm{true}}(\theta)}{w_{\mathrm{model}}(\theta)}
=
\frac{
\int dz\,\left\langle n^2(z,\theta)\right\rangle_\theta\,\xi_m(\theta;z)
}{
\int dz\,\bar n^2(z)\,\xi_m(\theta;z)
}.
$$

---

# Experiment

> **Target:** Build an analytical framework to estimate the underestimation of clustering amplitude given $ n(z,\theta) $ and $ \bar n(z) $.

Let's set up the context first: the footprint is a Healpix map, and we have the redshift distribution $ n(z,\theta) $ in each pixel. We also have the relative number density in each pixel. This enables to calculate the global redshift distribution $\bar n(z)$. Note, the relative number density is only used to calculate $\bar n(z)$. We do not care about the impact of density variations on clustering amplitude.

We can directly calculate the 2PCF $\left\langle \delta n^2(z,\theta)\right\rangle_\theta$ and therefore get $\Delta w(\theta)$.


> **Instruction:** We want to implement the above using Python. The code should have an API taking outside inputs $n(z,\theta)$ and $\bar n(z)$, and returning $\Delta w(\theta)$.
1. Create a new directory `spatial_variation/enhance` and implement the above using Python. 
2. Refer to the code in `spatial_variation/`, especially `spatial_variation/analytical_calculation_clean.py` and `spatial_variation/analytical_calculation.ipynb`. Use `Pyccl` for matter power spectrum. 
3. Make a test using toy variations. Instead of using correlated variations, use random correlations with fix mean and varying variance. 
4. Print the enhancement factor: the mean inverse width of local $n(z,\theta)$ over the inverse width of global $n(z)$, 
$$
\left<\sigma_{\rm local}^{-1}\right>/\sigma_{\rm global}^{-1},~\sigma^{-1}=\int dz n^2(z).
$$
5. Make plots into /output: 1. $n(z,\theta)$ and $\bar n(z)$, 2. $\Delta w(\theta)$, 3. $w_{\mathrm{true}}(\theta)$ versus $w_{\mathrm{model}}(\theta)$.
6. **Sanity check 1:** when separation is 0, $\left\langle \delta n^2(z,\theta)\right\rangle_\theta$ as a 2PCF at $\theta=0$ should be equavelent to $\left\langle \delta n^2(z,\theta)\right\rangle_\theta$ as a mean. Calculate $\Delta w(\theta)$ both ways and print the comparison.
7. **Sanity check 2:** w(\theta) is approximately proportional to inverse of n(z) width when the kernel is narrow (matter power spectrum stays constant). Therefore, we can expect the clustering enhancement to be approximately euqal to the enhancement factor defined above (at least if we construct the local n(z) by only perturbing the mean of $\bar n(z)$).

> **Tests:** 
1. Make a plot of $\left\langle \delta n^2(z,\theta)\right\rangle_\theta$ as a function of $\theta$ and $z$. This shows the variation in terms of $z$ and angular separation. The former matters considering matter power spectrum evolves. Thus the clustering enhancement deviates from the enhancement factor defined above. 


## Condition-dependent tomographic bin assignment $P_i$

For a galaxy $g$ with true redshift $z_g$, intrinsic magnitude $m_g$, and sky position $\theta_g$, define the tomographic bin membership probability
$$
P_i(g\mid \mathbf{c}(\theta_g))
\equiv
\mathbb{P}\!\left(\hat z \in [z_i^{\min}, z_i^{\max}]
\;\big|\; z_g, m_g, \mathbf{c}(\theta_g)\right),
$$
where $\mathbf{c}(\theta)$ denotes observing conditions.

---

### Photo-$z$ likelihood model
Assume a Gaussian photo-$z$ error model
$$
\hat z = z_g + b(z_g,m_g,\mathbf{c}) + \epsilon,
\qquad
\epsilon \sim \mathcal{N}(0,\sigma_z^2).
$$

**Bias**
$$
b(z,m,\mathbf{c})
=
b_0 + b_1 z + b_m(m-m_{\rm ref}) + b_c\,T(\mathbf{c}).
$$

**Scatter (conditions enter mainly here)**
$$
\sigma_z(z,m,\mathbf{c})
=
\sigma_0(1+z)
\left[1+\alpha\big(m+\Delta m(\mathbf{c})-m_{\rm ref}\big)\right].
$$

A practical choice for the condition term is
$$
\Delta m(\mathbf{c}(\theta)) = m_{\rm lim,ref} - m_{\rm lim}(\theta),
$$
so worse depth increases $\sigma_z$.

---

### Bin membership probability
For bin $i=[z_i^{\min}, z_i^{\max}]$,
$$
P_i(g\mid \mathbf{c}(\theta_g))
=
\Phi\!\left(\frac{z_i^{\max}-\mu_g}{\sigma_g}\right)
-
\Phi\!\left(\frac{z_i^{\min}-\mu_g}{\sigma_g}\right),
$$
with
$$
\mu_g = z_g + b(z_g,m_g,\mathbf{c}(\theta_g)),
\qquad
\sigma_g = \sigma_z(z_g,m_g,\mathbf{c}(\theta_g)).
$$

---

### Use in tomographic $n_i(z,\theta)$
The expected (unnormalized) per-pixel bin distribution is
$$
\tilde n_i(z,\theta)
=
\sum_{g\in \theta}
p_{\rm det}\!\left(\mathbf{c}(\theta),\mathrm{props}_g\right)\,
P_i(g\mid \mathbf{c}(\theta))\,
\delta_D(z-z_g),
$$
which can be normalized per pixel to define $n_i(z,\theta)$.

**Key point:** observing conditions affect binning through  
(i) detection probability $p_{\rm det}$ (sample composition) and  
(ii) $P_i$ via condition-dependent photo-$z$ scatter (and optionally bias).

