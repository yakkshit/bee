# 📖 Scientific Documentation, Bioassay Protocol & Mathematical Foundations

Dedicated documentation for **Antigravity Bee Analytics Studio** — an open-source, publication-grade computational biostatistics platform designed for honeybee (*Apis mellifera*) toxicology, longevity kinetics, oral exposure bioassays, and nutritional feeding persistence.

---

## 📌 Table of Contents
- [1. Quick Start & Execution](#1-quick-start--execution)
- [2. Core Bioassay Protocol Rules](#2-core-bioassay-protocol-rules)
  - [Rule 1: Post-2h Starvation Rule (Exclusion of Time Zero / $d_0$)](#rule-1-post-2h-starvation-rule-exclusion-of-time-zero--d_0)
  - [Rule 2: Standard 4 Experimental Cohorts & Normalized Naming](#rule-2-standard-4-experimental-cohorts--normalized-naming)
  - [Rule 3: Resurrected Bee Observation Correction Rule](#rule-3-resurrected-bee-observation-correction-rule)
- [3. Mathematical Formulations & Statistical Methods](#3-mathematical-formulations--statistical-methods)
  - [3.1 Logistic Sigmoidal Growth Curve](#31-logistic-sigmoidal-growth-curve)
  - [3.2 Finney Probit Regression & Abbott's Formula](#32-finney-probit-regression--abbotts-formula)
  - [3.3 Lethal Time Estimation ($LT_{10}$–$LT_{90}$) & Mathematical Boundary Guards](#33-lethal-time-estimation-lt_10lt_90--mathematical-boundary-guards)
  - [3.4 Kaplan–Meier Product-Limit Estimator & Greenwood Variance](#34-kaplanmeier-product-limit-estimator--greenwood-variance)
  - [3.5 Mantel–Cox Log-Rank Test & Cumulative Hazard Curves](#35-mantelcox-log-rank-test--cumulative-hazard-curves)
  - [3.6 Cox Proportional Hazards Regression & Forest Plot](#36-cox-proportional-hazards-regression--forest-plot)
  - [3.7 Daily & Cumulative Feeding Kinetics (Sucrose `Cons` vs. Treatment `TCons`)](#37-daily--cumulative-feeding-kinetics-sucrose-cons-vs-treatment-tcons)
- [4. Software Architecture & Code Snippets](#4-software-architecture--code-snippets)
- [5. Repository Hygiene: Purging Unwanted Files on GitHub](#5-repository-hygiene-purging-unwanted-files-on-github)
- [6. Academic Citation](#6-academic-citation)

---

## 1. Quick Start & Execution

### 1.1 Prerequisites
- Python 3.9, 3.10, or 3.11
- Recommended OS: macOS, Linux, or Windows (WSL2 / PowerShell)

### 1.2 Installation & Launch
```bash
# 1. Clone repository
git clone https://github.com/yakkshit/bee.git
cd bee

# 2. Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Launch Streamlit Studio
streamlit run streamlit_app.py --server.port 8501
```

Once running:
- Main Dashboard: `http://localhost:8501`
- Dedicated Route: `http://localhost:8501/doc` or `http://localhost:8501/?page=doc`

---

## 2. Core Bioassay Protocol Rules

To ensure reproducibility across laboratories and prevent observational noise from skewing parametric fits, the engine strictly enforces three foundational protocol rules:

### Rule 1: Post-2h Starvation Rule (Exclusion of Time Zero / $d_0$)
- **Biological Context**: Prior to treatment exposure, bees are subjected to a standardized **2-hour starvation fasting window**. This synchronizes their hunger status and metabolic baseline.
- **Strict Exclusion**: Data logged at time zero ($t=0$, `d0`, `Day 0`) represents pre-starvation handling and cage setup. **Time zero ($t=0$) is strictly excluded from all survival curves, probit regressions, lethal time calculations, Kaplan-Meier product-limit curves, and hazard ratios.**
- **Origin of Assay**: Day 1 ($d_1 = 1.0$ elapsed day post-starvation) marks the true biological start of treatment exposure.

### Rule 2: Standard 4 Experimental Cohorts & Normalized Naming
To maintain consistent cross-cohort comparison and eliminate ad-hoc file naming artifacts, datasets are mapped to 4 agreed standard cohorts:
1. **`sucrose solution 2 microliter`** (Code Name: `Sucrose 2 µL` | Control Baseline $HR \equiv 1.0$)
2. **`5 % acetone 2 microliter`** (Code Name: `5% Acetone 2 µL`)
3. **`10 % acetone 2 microliter`** (Code Name: `10% Acetone 2 µL`)
4. **`5 % acetone 5 microliter`** (Code Name: `5% Acetone 5 µL`)

*Blinded Assay Mode*: In initial blinded trials when treatments are unknown, files are assigned to `None / Blinded` and clean file identifiers are displayed.

### Rule 3: Resurrected Bee Observation Correction Rule
- In field assays, individual bees may be temporarily comatose or immobile, causing human experimenters to log them as Dead (`D`) on Day $k$. If the same bee subsequently displays movement or antennal reflexes (`A` or `B`) on Day $j > k$, the later alive status takes precedence.
- An individual bee is only classified as deceased at day $d^*$ if and only if **all subsequent observations through the end of the experiment remain Dead (`D`)**.

---

## 3. Mathematical Formulations & Statistical Methods

### 3.1 Logistic Sigmoidal Growth Curve

Cumulative mortality fraction $P(t) \in [0, 1]$ over elapsed days $t \ge 1$ is modeled using the continuous 3-parameter logistic function:
$$P(t) = \frac{L}{1 + \exp\left(-k(t - t_0)\right)}$$

- $L \in (0, 1]$: Upper asymptote representing maximum cumulative mortality.
- $k > 0$: Growth rate constant / steepness parameter ($\text{days}^{-1}$).
- $t_0$: Inflection point (time of maximum mortality rate, where $P(t_0) = L/2$).

**Biological Rationale**: Biological organisms exhibit a bell-shaped tolerance distribution to toxic chemical exposure. The cumulative response across an entire population thus follows an S-shaped sigmoid curve.

---

### 3.2 Finney Probit Regression & Abbott's Formula

#### Abbott's Natural Mortality Correction
To account for baseline control mortality:
$$P_{\text{corrected}} = \frac{P_{\text{treatment}} - P_{\text{control}}}{1 - P_{\text{control}}}$$
If $P_{\text{treatment}} \le P_{\text{control}}$, $P_{\text{corrected}} = 0$.

#### Finney Probit Linearization
The cumulative response probability $P$ is converted to standard normal quantiles shifted by 5:
$$\text{Probit}(P) = \Phi^{-1}(P) + 5$$
where $\Phi(z) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^z e^{-u^2/2} \, du$.

Linear model against logarithmic time $x = \log_{10}(t)$:
$$Y = a + b \cdot \log_{10}(t)$$
Fitted using iteratively reweighted least squares (IRLS) with Finney weights:
$$w_i = \frac{Z_i^2}{P_i(1 - P_i)}, \quad Z_i = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{(\text{Probit}(P_i)-5)^2}{2}\right)$$

#### Chi-Square Goodness-of-Fit
$$\chi^2 = \sum_{i=1}^k \frac{(O_i - E_i)^2}{E_i}, \quad df = k - 2$$
A $p$-value $> 0.05$ demonstrates that the observed data does not significantly deviate from the linear probit model.

---

### 3.3 Lethal Time Estimation ($LT_{10}$–$LT_{90}$) & Mathematical Boundary Guards

#### Analytical Solution
To determine the time required to kill $X\%$ of the population ($LT_X$):
$$Y_X = \Phi^{-1}(X / 100) + 5$$
Solving the linear equation for $t$:
$$\log_{10}(LT_X) = \frac{Y_X - a}{b} \implies LT_X = 10^{\frac{Y_X - a}{b}}$$

- $LT_{50}$ (Median Lethal Time): $Y_{50} = 5.000 \implies LT_{50} = 10^{(5 - a)/b}$
- $LT_{10}$ (10% Lethality): $Y_{10} = 3.718 \implies LT_{10} = 10^{(3.718 - a)/b}$
- $LT_{90}$ (90% Lethality): $Y_{90} = 6.282 \implies LT_{90} = 10^{(6.282 - a)/b}$

#### Boundary Guards Against Mathematical Artifacts
When a cohort has minimal or zero mortality (such as untreated control groups), the empirical slope $b \to 0$. In pure mathematical inversion, dividing by a near-zero slope produces absurd numbers ($24,893\text{ days}$).
The studio applies the following boundary guards:
1. **Slope Cutoff**: If $b \le 0.01$, $LT_X$ is defined as `Indeterminate / Undefined` ($\infty$) with an explanatory warning.
2. **Horizon Clipping**: If calculated $LT_{50} > 3 \times t_{\max}$ (typically $3 \times 15 = 45\text{ days}$), the value is capped at $> t_{\max}$ with a diagnostic alert.

---

### 3.4 Kaplan–Meier Product-Limit Estimator & Greenwood Variance

Non-parametric cumulative survival probability at observation time $t$:
$$\hat{S}(t) = \prod_{t_i \le t} \left(1 - \frac{d_i}{n_i}\right)$$
- $t_i$: Days with one or more confirmed deaths.
- $d_i$: Number of deaths at time $t_i$.
- $n_i$: Number of living bees at risk right before $t_i$.

#### Greenwood's Formula for Standard Error
$$\widehat{\text{Var}}(\hat{S}(t)) = \hat{S}(t)^2 \sum_{t_i \le t} \frac{d_i}{n_i(n_i - d_i)}$$

Log-log transformation 95% confidence intervals:
$$\hat{S}(t)^{\exp\left(\pm 1.96 \cdot \frac{\sqrt{\widehat{\text{Var}}(\hat{S}(t))}}{\hat{S}(t) \ln \hat{S}(t)}\right)}$$

---

### 3.5 Mantel–Cox Log-Rank Test & Cumulative Hazard Curves

To test the null hypothesis $H_0: S_1(t) = S_2(t) = \dots = S_K(t)$:
1. At each distinct event time $t_i$, expected deaths for cohort $j$ are:
   $$E_{ji} = n_{ji} \frac{d_i}{n_i}$$
2. Summing over all time points yields total observed $O_j = \sum_i d_{ji}$ and total expected $E_j = \sum_i E_{ji}$.
3. Log-Rank test statistic:
   $$\chi^2 = \sum_{j=1}^K \frac{(O_j - E_j)^2}{E_j} \sim \chi^2(K - 1)$$

#### Cumulative Hazard Step Curves
Nelson-Aalen estimator:
$$H(t) = -\ln \hat{S}(t) = \sum_{t_i \le t} \frac{d_i}{n_i}$$
A steeper step trajectory represents higher instantaneous mortality pressure.

---

### 3.6 Cox Proportional Hazards Regression & Forest Plot

The semi-parametric hazard rate for an individual subject given covariates $\mathbf{X}$ is:
$$h(t \mid \mathbf{X}) = h_0(t) \exp\left(\sum_{j=1}^p \beta_j X_j\right)$$
- $h_0(t)$: Non-parametric baseline hazard.
- Hazard Ratio ($HR$):
  $$HR_j = \exp(\beta_j)$$
  Compared against `Sucrose 2 µL` baseline ($HR \equiv 1.0$).
- 95% Confidence Interval:
  $$\exp\left(\hat{\beta}_j \pm 1.96 \cdot \text{SE}(\hat{\beta}_j)\right)$$

---

### 3.7 Daily & Cumulative Feeding Kinetics (Sucrose `Cons` vs. Treatment `TCons`)

Daily intake per surviving bee ($\mu\text{L} / \text{bee} / \text{day}$):
$$\text{Consumption}_d = \frac{V_{d-1} - V_d - V_{\text{evap}}}{N_{\text{alive}, d}}$$
- $V_d$: Measured feeder volume on Day $d$ ($\mu\text{L}$).
- $V_{\text{evap}}$: Evaporative correction determined from an unpopulated control cage.
- $N_{\text{alive}, d}$: Count of active, living bees on Day $d$.

The studio enables dual intake comparison:
- **`Cons` (Sucrose Intake)**: Caloric sustenance intake.
- **`TCons` (Treatment Intake)**: Ingested volume of test treatment solution.
- **Standard Error of Mean (SEM)**: $\text{SEM} = \sigma / \sqrt{n}$.

---

## 4. Software Architecture & Code Snippets

### Probit Estimation in Python
```python
import numpy as np
from modules.core.probit_engine import ProbitEngine

# Days strictly post-2h starvation: Day 1 onwards
days = np.arange(1, 16)
dead_counts = np.array([0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])
total_bees = 10

engine = ProbitEngine(days, dead_counts, total_bees)
results = engine.fit()

print(f"Slope (b): {results['slope']:.4f}")
print(f"LT50: {results['LT50']:.2f} Days")
print(f"Chi-Square: {results['chi2']:.3f} (p = {results['p_value']:.4f})")
```

---

## 5. Repository Hygiene: Purging Unwanted Files on GitHub

### Why Unwanted Files (e.g. `.venv_run/`) Ended Up in GitHub
When `git add .` was executed before `.gitignore` was configured, Git added the entire virtual environment (`.venv_run/`, >10,000 files) into the tracked Git index.
Simply adding `.venv_run/` to `.gitignore` **does not untrack already committed files**.

### Step-by-Step Fix: Safely Untrack Without Deleting Local Files
Run these commands in your local project root:

```bash
# 1. Untrack the virtual environment from Git without deleting files from disk:
git rm -r --cached .venv_run/

# 2. Untrack any agent metadata directories:
git rm -r --cached .agents/ .claude/ 2>/dev/null || true

# 3. Commit the changes:
git commit -m "chore: purge committed virtual environment from git repository"

# 4. Push to GitHub main branch:
git push origin main
```

**Why this works**:
- The `--cached` flag instructs Git to delete the file references only from the Git index (staging tree), leaving your local Python virtual environment, dependencies, and local data completely intact.

---

## 6. Academic Citation

If you use this software in your research or publication, please cite as:

```bibtex
@software{bee_analytics_studio_2026,
  author = {Antigravity Bioassay Consortium},
  title = {Antigravity Bee Analytics Studio: A Publication-Grade Computational Biostatistical Suite for Honeybee (Apis mellifera) Bioassays},
  year = {2026},
  url = {https://github.com/yakkshit/bee}
}
```

