"""
Scientific Documentation, Bioassay Protocol & Mathematical Foundations
Dedicated documentation module for  Bee Analytics Studio.
Designed for peer-reviewed publication reproducibility, statistical transparency, and open-science compliance.
"""

import streamlit as st

def render_doc_page():
    st.markdown("""
    # 📖 Scientific Documentation & Bioassay Protocol

    Welcome to the comprehensive technical documentation for the **Bee Analytics Studio**.
    This guide provides the complete experimental, statistical, and algorithmic framework for honeybee (*Apis mellifera*) oral exposure bioassays, designed for peer-reviewed publication and open-science reproducibility.

    ---
    """)

    # Quick Jump Navigation Bar
    st.info("""
    **Quick Navigation:**
    - [1. Quick Start & Execution](#1-quick-start--execution)
    - [2. Core Bioassay Protocol Rules](#2-core-bioassay-protocol-rules)
    - [3. Mathematical Formulations & Statistical Methods (Plot-by-Plot)](#3-mathematical-formulations--statistical-methods)
      - [3.1 Logistic Sigmoidal Growth Curve](#31-logistic-sigmoidal-growth-curve)
      - [3.2 Finney Probit Regression & Abbott's Formula](#32-finney-probit-regression--abbotts-formula)
      - [3.3 Lethal Time Estimation ($LT_{10}$–$LT_{90}$) & Boundary Guards](#33-lethal-time-estimation-lt10lt90--boundary-guards)
      - [3.4 Kaplan–Meier Survival Estimator & Greenwood Variance](#34-kaplanmeier-survival-estimator--greenwood-variance)
      - [3.5 Mantel–Cox Log-Rank Test & Cumulative Hazard Curves](#35-mantelcox-log-rank-test--cumulative-hazard-curves)
      - [3.6 Cox Proportional Hazards Regression & Forest Plot](#36-cox-proportional-hazards-regression--forest-plot)
      - [3.7 Sucrose vs. Treatment Solution Consumption Dynamics](#37-sucrose-vs-treatment-solution-consumption-dynamics)
    - [4. Code Snippets & Reproducibility](#4-code-snippets--reproducibility)
    - [5. Repository Hygiene: Untracking Unnecessary Files on GitHub](#5-repository-hygiene-untracking-unnecessary-files-on-github)
    - [6. Academic Citation](#6-academic-citation)
    """)

    # Section 1: Quick Start
    st.markdown("""
    ## 1. Quick Start & Execution

    ### 1.1 System Requirements
    - **Python**: 3.9, 3.10, or 3.11 (tested on macOS, Linux, and Windows)
    - **Key Libraries**: Streamlit ($\ge 1.32$), Lifelines ($\ge 0.28$), SciPy, Statsmodels, Matplotlib, Pandas, NumPy.

    ### 1.2 Running the Web Application
    ```bash
    # 1. Clone repository
    git clone https://github.com/yakkshit/bee.git
    cd bee

    # 2. Create and activate a clean virtual environment
    python3.11 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

    # 3. Install dependencies
    pip install -r requirements.txt

    # 4. Launch the Bee Analytics Studio
    streamlit run streamlit_app.py --server.port 8501
    ```
    Once launched, open your web browser to `http://localhost:8501` or access this documentation directly at `http://localhost:8501/?page=doc` or `http://localhost:8501/doc`.

    ---
    ## 2. Core Bioassay Protocol Rules

    The computational engine enforces three immutable bioassay protocol rules to guarantee valid biological inferences and prevent observational artifacts from corrupting regression models.

    ### Rule 1: Post-2h Starvation Rule (Exclusion of Time Zero / $d_0$)
    - **Experimental Background**: Honeybees collected from host hives undergo a mandatory **2-hour pre-experimental starvation period** to induce uniform feeding motivation and synchronize their metabolic baseline.
    - **Mathematical Requirement**: Pre-starvation baseline measurements ($t = 0$, labeled as `d0` or `Day 0`) reflect cage-handling acclimation and baseline setup. **Time zero ($t=0$) is strictly excluded from all survival curves, probit regressions, lethal time calculations, Kaplan-Meier estimations, and log-rank statistics.**
    - **First Evaluation Day**: Day 1 ($d_1$, representing 1.0 day of exposure post-starvation) is the true origin of treatment exposure.

    ### Rule 2: Standard 4 Experimental Cohorts & Normalized Naming
    To prevent ambiguity and maintain rigorous multi-laboratory consistency, raw files and messy headers are mapped to the four standard experimental groups:
    1. **`sucrose solution 2 microliter`** (Code Name: `Sucrose 2 µL` | Baseline Control, Hazard Ratio $HR \equiv 1.0$)
    2. **`5 % acetone 2 microliter`** (Code Name: `5% Acetone 2 µL`)
    3. **`10 % acetone 2 microliter`** (Code Name: `10% Acetone 2 µL`)
    4. **`5 % acetone 5 microliter`** (Code Name: `5% Acetone 5 µL`)

    *Blinded Support*: During initial blind phases, when treatments are unknown, cohorts are marked as `None / Blinded` and raw file identifiers are used until unblinded.

    ### Rule 3: Resurrected Bee Observation Correction Rule
    In field bioassays, individual bees may be knocked down, lethargic, or immobile, causing human observers to log them as Dead (`D`) on Day $k$. If the same bee subsequently recovers mobility or antennal response (`A` or `B`) on Day $j > k$, the later alive status takes absolute precedence.
    $$\text{Status}(i, d) = \begin{cases} \text{Alive}, & \text{if } \exists j \ge d \text{ such that observation is } A \text{ or } B \\ \text{Dead}, & \text{otherwise} \end{cases}$$
    A bee is only treated as deceased at day $d^*$ if and only if **all subsequent observations through the end of the bioassay remain Dead (`D`)**.

    ---
    ## 3. Mathematical Formulations & Statistical Methods

    Below is the complete mathematical derivation and scientific justification for each analytical module and figure generated by the studio.
    """)

    # 3.1 Logistic Model
    st.markdown(r"""
    ### 3.1 Logistic Sigmoidal Growth Curve

    #### Mathematical Model
    The cumulative mortality fraction $P(t) \in [0, 1]$ over elapsed observation days $t \ge 1$ is modeled using the 3-parameter continuous logistic function:
    $$P(t) = \frac{L}{1 + \exp\left(-k(t - t_0)\right)}$$

    - $L \in (0, 1]$: Upper asymptote denoting maximum cumulative mortality.
    - $k > 0$: Steepness coefficient / intrinsic mortality acceleration rate ($\text{days}^{-1}$).
    - $t_0$: Sigmoidal inflection point (day of peak daily mortality rate, where $P(t_0) = L/2$).

    #### Biological Rationale
    Toxicity kinetics follow biological tolerance distributions. Early in exposure, robust bees withstand toxin accumulation; as physiological thresholds are breached, mortality surges around $t_0$, and finally plateaus among resistant individuals.

    ---
    """)

    # 3.2 Finney Probit Regression
    st.markdown(r"""
    ### 3.2 Finney Probit Regression & Abbott's Formula

    #### Abbott's Natural Mortality Correction
    When background mortality occurs in the negative control group (`Sucrose 2 µL`), treatment mortality must be corrected to isolate treatment-induced toxicity:
    $$P_{\text{corrected}} = \frac{P_{\text{treatment}} - P_{\text{control}}}{1 - P_{\text{control}}}$$
    If $P_{\text{treatment}} \le P_{\text{control}}$, $P_{\text{corrected}}$ is constrained to zero.

    #### Finney Probit Linearization
    The Probit transformation maps cumulative response probabilities onto standard normal deviates (quantiles) with an added constant of 5 (to avoid negative values, following Finney 1971):
    $$\text{Probit}(P) = \Phi^{-1}(P) + 5$$
    where $\Phi(z) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^z e^{-u^2/2} \, du$ is the cumulative distribution function (CDF) of the standard normal distribution $\mathcal{N}(0, 1)$.

    The linearized equation against logarithmic exposure time $x = \log_{10}(t)$ is:
    $$Y = a + b \cdot \log_{10}(t)$$
    Estimated via weighted ordinary least squares (OLS) with empirical probit weighting:
    $$w_i = \frac{Z_i^2}{P_i (1 - P_i)}, \quad Z_i = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{(\text{Probit}(P_i)-5)^2}{2}\right)$$

    #### Goodness-of-Fit $\chi^2$ Test
    $$\chi^2 = \sum_{i=1}^k \frac{(O_i - E_i)^2}{E_i}, \quad df = k - 2$$
    A $p > 0.05$ confirms that the linear probit model adequately fits the empirical bioassay data.
    ---
    """)

    # 3.3 Lethal Time Estimation
    st.markdown(r"""
    ### 3.3 Lethal Time Estimation ($LT_{10}$–$LT_{90}$) & Boundary Guards

    #### Inversion Formula
    To calculate the lethal time required to eliminate $X\%$ of the cohort ($LT_X$, e.g., $LT_{10}, LT_{50}, LT_{90}$):
    1. Determine the target probit value: $Y_X = \Phi^{-1}(X / 100) + 5$
       - For $LT_{50}$: $P = 0.50 \implies \Phi^{-1}(0.50) = 0 \implies Y_{50} = 5.000$
       - For $LT_{10}$: $P = 0.10 \implies \Phi^{-1}(0.10) = -1.282 \implies Y_{10} = 3.718$
       - For $LT_{90}$: $P = 0.90 \implies \Phi^{-1}(0.90) = +1.282 \implies Y_{90} = 6.282$
    2. Solve for elapsed time $t$:
       $$\log_{10}(LT_X) = \frac{Y_X - a}{b} \implies LT_X = 10^{\frac{Y_X - a}{b}}$$

    #### Mathematical Boundary Guards (Preventing Infinite/Anomalous Extrapolations)
    In cohorts with very low mortality (e.g. baseline sucrose controls or very weak concentrations), the slope $b$ can approach zero, producing astronomically large, biologically impossible numbers (such as $24,893\text{ days}$).
    The software applies rigorous boundary guards:
    - **Flat or Negative Slope Guard**: If slope $b \le 0.01$, $LT_X$ is marked as `Indeterminate / Undefined` ($\infty$).
    - **Extrapolation Limit**: If calculated $LT_X > 3 \times t_{\max}$ (where $t_{\max}$ is the experimental duration, typically 15 days), the value is capped at $> t_{\max}$ with a diagnostic warning rather than distorting bar chart axes.
    ---
    """)

    # 3.4 Kaplan-Meier
    st.markdown(r"""
    ### 3.4 Kaplan–Meier Survival Estimator & Greenwood Variance

    #### Non-Parametric Product-Limit Estimator
    Survival probability $S(t) = P(T > t)$ at any observation time $t$ is calculated without assuming a parametric distribution:
    $$\hat{S}(t) = \prod_{t_i \le t} \left(1 - \frac{d_i}{n_i}\right)$$
    - $t_i$: Distinct observation days where at least one death occurred.
    - $d_i$: Number of confirmed deaths at time $t_i$.
    - $n_i$: Number of individuals alive and at risk immediately prior to $t_i$.

    #### Greenwood's Formula for Standard Error
    $$\widehat{\text{Var}}(\hat{S}(t)) = \hat{S}(t)^2 \sum_{t_i \le t} \frac{d_i}{n_i(n_i - d_i)}$$
    The 95% confidence bands are computed using the log-log transformation to guarantee bounds within $[0, 1]$:
    $$\ln(-\ln \hat{S}(t)) \pm 1.96 \cdot \frac{\sqrt{\sum \frac{d_i}{n_i(n_i - d_i)}}}{\ln \hat{S}(t)}$$
    ---
    """)

    # 3.5 Mantel-Cox Log-Rank Test
    st.markdown(r"""
    ### 3.5 Mantel–Cox Log-Rank Test & Cumulative Hazard Curves

    #### Hypothesis Testing
    To test whether survival distributions differ significantly across cohorts:
    - **Null Hypothesis ($H_0$)**: $S_1(t) = S_2(t) = \dots = S_K(t)$ across all $K$ cohorts.
    - **Alternative Hypothesis ($H_1$)**: At least two cohorts exhibit unequal survival curves.

    #### Expected Events & Chi-Square Statistic
    At each event time $t_i$, the expected number of deaths in cohort $j$ under $H_0$ is:
    $$E_{ji} = n_{ji} \left(\frac{d_i}{n_i}\right)$$
    Summing across all time points yields total observed $O_j = \sum_i d_{ji}$ and total expected $E_j = \sum_i E_{ji}$.
    The Log-Rank statistic follows a chi-square distribution with $K - 1$ degrees of freedom:
    $$\chi^2 = \sum_{j=1}^K \frac{(O_j - E_j)^2}{E_j} \sim \chi^2(K - 1)$$

    #### Cumulative Hazard Step Curves $H(t)$
    The Nelson–Aalen cumulative hazard $H(t) = -\ln \hat{S}(t)$ directly quantifies the cumulative force of mortality:
    $$H(t) = \sum_{t_i \le t} \frac{d_i}{n_i}$$
    A steeper slope on the cumulative hazard step curve indicates higher instantaneous mortality pressure on that treatment cohort.
    ---
    """)

    # 3.6 Cox Proportional Hazards
    st.markdown(r"""
    ### 3.6 Cox Proportional Hazards Regression & Forest Plot

    #### Semi-Parametric Formulation
    The hazard rate of death at time $t$ given experimental covariate vector $\mathbf{X}$ is:
    $$h(t \mid \mathbf{X}) = h_0(t) \exp\left(\sum_{j=1}^p \beta_j X_j\right)$$
    - $h_0(t)$: Unspecified, non-parametric baseline hazard function.
    - $\beta_j$: Regression coefficient representing the log-hazard impact of cohort $j$.

    #### Hazard Ratio ($HR$)
    Comparing cohort $j$ against the baseline control (`Sucrose 2 µL`):
    $$HR_j = \exp(\beta_j)$$
    - $HR = 1.0$: Baseline control mortality risk.
    - $HR > 1.0$: Increased mortality hazard (toxic acceleration).
    - $HR < 1.0$: Decreased mortality hazard (protective effect).

    #### Forest Plot Representation
    The Forest Plot displays estimated $HR$ values as center points (diamonds for reference baseline $HR=1.0$, circles for treatments) with horizontal error bars indicating 95% Wald Confidence Intervals:
    $$95\% \text{ CI} = \left[\exp\left(\hat{\beta}_j - 1.96 \cdot \text{SE}(\hat{\beta}_j)\right), \, \exp\left(\hat{\beta}_j + 1.96 \cdot \text{SE}(\hat{\beta}_j)\right)\right]$$
    ---
    """)

    # 3.7 Consumption Dynamics
    st.markdown(r"""
    ### 3.7 Sucrose vs. Treatment Solution Consumption Dynamics

    #### Measurement Formula
    Daily volumetric consumption per living bee ($\mu\text{L} / \text{bee} / \text{day}$) is calculated by tracking daily feeder meniscus levels:
    $$\text{Consumption}_d = \frac{V_{d-1} - V_d - V_{\text{evaporation}}}{N_{\text{alive}, d}}$$
    - $V_d$: Measured liquid level in the feeder on Day $d$ ($\mu\text{L}$).
    - $V_{\text{evaporation}}$: Evaporative loss determined from an unpopulated control cage.
    - $N_{\text{alive}, d}$: Count of surviving, active bees in the cage on Day $d$.

    #### Pure Sucrose (`Cons`) vs. Treatment Solution (`TCons`)
    Bioassays evaluate both general nutritional appetite and acute toxicant solution intake:
    1. **`Cons` (Sucrose Intake)**: Baseline caloric feeding from pure 50% w/w sucrose solution.
    2. **`TCons` (Treatment Intake)**: Consumption of the test chemical/acetone solution.
    3. **Nutritional Anorexia Analysis**: Reductions in `Cons` relative to control indicate feeding deterrent effects, malaise, or neurotoxic suppression of appetite.
    ---
    """)

    # Section 4: Code Snippets
    st.markdown("""
    ## 4. Code Snippets & Reproducibility

    Researchers can run the exact statistical engine directly via standalone Python scripts or in automated pipelines.

    ### 4.1 Running Probit & Lethal Time Calculation
    ```python
    import numpy as np
    from modules.core.probit_engine import ProbitEngine

    # Observation days (strictly post-2h starvation: Day 1 onwards)
    days = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    # Cumulative dead bees count out of 10 total
    dead_counts = np.array([0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])
    total_bees = 10

    # Initialize Finney probit engine
    engine = ProbitEngine(days, dead_counts, total_bees)
    fit_results = engine.fit()

    print(f"Probit Slope (b): {fit_results['slope']:.4f}")
    print(f"Probit Intercept (a): {fit_results['intercept']:.4f}")
    print(f"Estimated LT50: {fit_results['LT50']:.2f} Days")
    print(f"Estimated LT90: {fit_results['LT90']:.2f} Days")
    print(f"Model Chi-Square: {fit_results['chi2']:.3f} (p = {fit_results['p_value']:.4f})")
    ```

    ### 4.2 Automated Batch Auditor
    ```python
    from modules.core.csv_auditor import audit_csv_files

    # Audit list of uploaded or local CSV files for bioassay protocol compliance
    csv_paths = ["FM_261027_02A.csv", "FM_261027_02B.csv", "FM_261027_02C.csv"]
    audit_results = audit_csv_files(csv_paths)

    for res in audit_results:
        print(f"File: {res['filename']} | Status: {res['status']}")
        for warning in res['warnings']:
            print(f"  ⚠️ Warning: {warning}")
    ```

    ---
    ## 5. Repository Hygiene: Untracking Unnecessary Files on GitHub

    ### Why Unwanted Files (e.g., `.venv_run/`) Ended Up in GitHub
    When a file or folder (such as a local virtual environment `.venv_run/` or temporary test files) is committed via `git add .` before it is listed in `.gitignore`, Git records it into its persistent tracking index.
    Adding `.venv_run/` to `.gitignore` **afterwards does NOT remove it from Git's tracking**. It only prevents *new* files from being added.

    ### Step-by-Step Fix: Removing Files Without Deleting Them Locally
    To remove the virtual environment and cached files from GitHub while safely keeping them on your computer:

    ```bash
    # 1. Open your terminal in the repository root
    cd /path/to/bee

    # 2. Untrack the virtual environment from git's staging index
    # (The --cached flag tells Git to remove tracking without deleting the folder on your hard drive)
    git rm -r --cached .venv_run/

    # 3. Untrack any other cached IDE or temp directories if needed
    git rm -r --cached .agents/ .claude/ 2>/dev/null || true

    # 4. Commit the removal
    git commit -m "chore: purge committed virtual environment and caches from repository"

    # 5. Push to GitHub main branch
    git push origin main
    ```

    Once completed, GitHub will no longer show `.venv_run` in the repository, reducing repository clone size from over 10,000 files to clean source code!

    ---
    ## 6. Academic Citation

    If you utilize  Bee Analytics Studio in your research or publication, please cite as:

    > ** Bee Analytics Studio**: A publication-grade computational biostatistical suite for honeybee (*Apis mellifera*) toxicology and feeding kinetics.  Bioassay Consortium (2026). Available at: `https://github.com/yakkshit/bee`.

    """)

if __name__ == "__main__":
    render_doc_page()

