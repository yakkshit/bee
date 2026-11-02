# 📐 Mathematical Models, Statistical Derivations & Plot Rationale

This document provides complete mathematical derivations, regression procedures, and statistical hypothesis tests used in **Bee Analytics Studio**.

---

## 1. Logistic Sigmoidal Growth Curve

### Equation
$$P(t) = \frac{L}{1 + \exp\left(-k(t - t_0)\right)}$$

- $P(t) \in [0, 1]$: Cumulative fraction of dead bees at elapsed day $t \ge 1.0$.
- $L \in (0, 1]$: Upper asymptote (theoretical carrying capacity / maximal cumulative mortality).
- $k > 0$: Growth steepness parameter ($\text{days}^{-1}$).
- $t_0$: Time of sigmoidal inflection point (point of maximum daily mortality rate, where $P(t_0) = L/2$).

### Estimation
Fitted via non-linear least squares (Levenberg-Marquardt optimization) minimizing the residual sum of squares:
$$\text{RSS} = \sum_{t=1}^N \left(P_{\text{obs}}(t) - P(t)\right)^2$$

---

## 2. Finney Probit Regression & Abbott Correction

### Abbott's Formula
To isolate chemical-specific mortality from natural background control mortality:
$$P_{\text{corrected}} = \frac{P_{\text{treatment}} - P_{\text{control}}}{1 - P_{\text{control}}}$$
If $P_{\text{treatment}} \le P_{\text{control}}$, $P_{\text{corrected}} = 0$.

### Probit Linearization
Cumulative probability $P$ is transformed to normal quantiles shifted by 5:
$$\text{Probit}(P) = \Phi^{-1}(P) + 5$$
where $\Phi(z) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^z e^{-u^2/2} \, du$.

Linear model against log-time $x = \log_{10}(t)$:
$$Y = a + b \cdot \log_{10}(t)$$
Weighting coefficients for iteratively reweighted least squares:
$$w_i = \frac{Z_i^2}{P_i(1 - P_i)}, \quad Z_i = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{(Y_i - 5)^2}{2}\right)$$

### Goodness-of-Fit
$$\chi^2 = \sum_{i=1}^k \frac{(O_i - E_i)^2}{E_i}, \quad df = k - 2$$

---

## 3. Lethal Time Estimation ($LT_{10} \dots LT_{90}$) & Boundary Guards

### Analytical Inversion
Target probit value for lethality fraction $X\%$:
$$Y_X = \Phi^{-1}(X / 100) + 5$$
Solving for time $t$:
$$\log_{10}(LT_X) = \frac{Y_X - a}{b} \implies LT_X = 10^{\frac{Y_X - a}{b}}$$

### Boundary Guards
- **Slope Boundary**: When a cohort exhibits little to no mortality (e.g. baseline control), $b \to 0$, causing $LT_{50} \to \infty$. If $b \le 0.01$, the calculation is guarded and reported as `Indeterminate` ($\infty$).
- **Time Horizon Ceiling**: Extrapolations beyond $3 \times t_{\max}$ are capped at $> t_{\max}$ with a diagnostic alert.

---

## 4. Kaplan–Meier Survival Estimator

### Product-Limit Formulation
$$\hat{S}(t) = \prod_{t_i \le t} \left(1 - \frac{d_i}{n_i}\right)$$
- $t_i$: Observation times with one or more events.
- $d_i$: Number of deaths at $t_i$.
- $n_i$: Number of individuals at risk right before $t_i$.

### Greenwood Variance & Confidence Bands
$$\widehat{\text{Var}}(\hat{S}(t)) = \hat{S}(t)^2 \sum_{t_i \le t} \frac{d_i}{n_i(n_i - d_i)}$$
95% log-log confidence intervals ensure bounds remain strictly within $[0, 1]$:
$$\ln(-\ln \hat{S}(t)) \pm 1.96 \cdot \frac{\sqrt{\sum \frac{d_i}{n_i(n_i - d_i)}}}{\ln \hat{S}(t)}$$

---

## 5. Mantel–Cox Log-Rank Test & Cumulative Hazard

### Null Hypothesis
$$H_0: S_1(t) = S_2(t) = \dots = S_K(t)$$

Expected deaths for cohort $j$ at time $t_i$:
$$E_{ji} = n_{ji} \frac{d_i}{n_i}$$
Log-Rank test statistic:
$$\chi^2 = \sum_{j=1}^K \frac{(O_j - E_j)^2}{E_j} \sim \chi^2(K - 1)$$

### Cumulative Hazard Step Curves
Nelson-Aalen cumulative hazard:
$$H(t) = -\ln \hat{S}(t) = \sum_{t_i \le t} \frac{d_i}{n_i}$$
Directly quantifies the cumulative force of mortality over time.

---

## 6. Cox Proportional Hazards Regression & Forest Plot

### Semi-Parametric Formulation
$$h(t \mid \mathbf{X}) = h_0(t) \exp\left(\sum_{j=1}^p \beta_j X_j\right)$$
- $h_0(t)$: Baseline hazard rate over time.
- Hazard Ratio: $HR_j = \exp(\beta_j)$ relative to `Sucrose 2 µL` baseline ($HR \equiv 1.0$).
- 95% Confidence Intervals: $\exp\left(\hat{\beta}_j \pm 1.96 \cdot \text{SE}(\hat{\beta}_j)\right)$.

---

## 7. Feeding Kinetics (Sucrose `Cons` vs. Treatment `TCons`)

Daily intake per live bee:
$$\text{Consumption}_d = \frac{V_{d-1} - V_d - V_{\text{evaporation}}}{N_{\text{alive}, d}} \quad (\mu\text{L} / \text{bee} / \text{day})$$
Standard Error of the Mean:
$$\text{SEM} = \frac{\sigma}{\sqrt{n}}$$
Allows discrimination between caloric feeding (`Cons`) and toxicant ingestion (`TCons`) to detect nutritional anorexia or feeding deterrence.

