"""
Scientific Documentation, Bioassay Protocol & Mathematical Foundations
Dedicated documentation module for Antigravity Bee Analytics Studio.
Designed for peer-reviewed publication reproducibility, statistical transparency, and open-science compliance.
"""

import os
import streamlit as st

def load_doc_file(filename: str) -> str:
    """Safely read documentation markdown file from doc/ directory."""
    doc_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "doc", filename)
    if os.path.exists(doc_path):
        with open(doc_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def render_doc_page():
    # Styled Hero Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 24px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 20px;">
        <h1 style="color: #f8fafc; margin: 0 0 8px 0; font-size: 2.2rem; font-weight: 700;">
            📖 Bee Analytics Studio — Scientific Documentation & Protocol
        </h1>
        <p style="color: #94a3b8; font-size: 1.05rem; margin: 0 0 14px 0; line-height: 1.5;">
            Publication-grade biostatistical reference, experimental ground rules, and mathematical formulations for honeybee (<em>Apis mellifera</em>) toxicological bioassays.
        </p>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <span style="background: #0284c7; color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">Python 3.11</span>
            <span style="background: #16a34a; color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">Streamlit 1.32</span>
            <span style="background: #d97706; color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">Post-2h Starvation Rule Compliant</span>
            <span style="background: #8b5cf6; color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">Lifelines 0.28</span>
            <span style="background: #475569; color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">Open Science / MIT</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Document Download Quick Action
    full_doc = load_doc_file("DOCUMENTATION.md")
    col_d1, col_d2 = st.columns([3, 1])
    with col_d1:
        st.caption("Browse interactive documentation sections below or download the standalone markdown manual for paper supplementary materials.")
    with col_d2:
        if full_doc:
            st.download_button(
                label="📥 Download Full Manual (.md)",
                data=full_doc,
                file_name="Bee_Analytics_Studio_Documentation.md",
                mime="text/markdown",
                use_container_width=True
            )

    # Interactive Navigation Tabs
    sub_tabs = st.tabs([
        "🚀 1. Quick Start & Execution",
        "🔬 2. Bioassay Protocol Rules",
        "📐 3. Mathematical Formulations (Plot-by-Plot)",
        "💻 4. Code Snippets & Pipelines",
        "🧹 5. Git Hygiene: Untracking Files",
        "📝 6. Citation & Export"
    ])

    # TAB 1: Quick Start
    with sub_tabs[0]:
        st.markdown("### 🚀 Quick Start & System Execution")
        st.markdown("""
        The **Antigravity Bee Analytics Studio** is designed for high-reproducibility toxicological research. Follow these steps to set up the environment and launch the software.
        """)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            #### ⚙️ Standard Local Execution
            ```bash
            # 1. Clone the repository
            git clone https://github.com/yakkshit/bee.git
            cd bee

            # 2. Create and activate a clean virtualenv
            python3.11 -m venv .venv
            source .venv/bin/activate

            # 3. Install dependencies
            pip install -r requirements.txt

            # 4. Launch the application
            streamlit run streamlit_app.py --server.port 8501
            ```
            """)
        with c2:
            st.markdown("""
            #### 🌐 Application Routes & Ports
            - **Main Analysis Studio**: `http://localhost:8501/`
            - **Direct Documentation Route**: `http://localhost:8501/doc` or `http://localhost:8501/?page=doc`
            - **Offline CLI Mode**: Probit engine and batch auditing can also be executed in headless scripts or HPC pipelines without a browser.
            """)

        st.info("💡 **Tip**: If running on macOS or Linux, ensure Python 3.11 is active (`python3.11 --version`). Dependencies in `requirements.txt` are pinned to prevent upstream breaking changes.")

    # TAB 2: Protocol Rules
    with sub_tabs[1]:
        st.markdown("### 🔬 Mandatory Bioassay Protocol Rules")
        st.markdown("""
        To guarantee biological validity and prevent experimental artifacts from skewing inferential statistics, the engine enforces three foundational protocol rules.
        """)

        with st.expander("⏱️ Rule 1: Post-2h Starvation Rule (Exclusion of Time Zero / Day 0)", expanded=True):
            st.markdown("""
            - **Biological Context**: Honeybees undergo a mandatory **2-hour pre-treatment starvation period**. This fasting period ensures that bees exhaust previous gut contents, synchronize hunger motivation, and feed promptly once exposure feeders are mounted.
            - **Strict Exclusion**: Data recorded at time zero ($t=0$, labeled as `d0` or `Day 0`) reflects cage handling, transport acclimation, and baseline equipment tare.
            - **Analytical Rule**: Time zero ($t=0$) is **strictly excluded** from all Probit linear regressions, Lethal Time estimations ($LT_{10} \dots LT_{90}$), Kaplan-Meier survival curves, Mantel-Cox log-rank tests, and Cox regression models.
            - **True Baseline**: Day 1 ($d_1 = 1.0\text{ day}$ post-starvation) is the true biological origin of treatment exposure.
            """)

        with st.expander("🏷️ Rule 2: Standard 4 Experimental Cohorts & Normalized Naming", expanded=True):
            st.markdown("""
            To eliminate ambiguous labels, dirty filenames, and cross-batch nomenclature discrepancies, all datasets map to four standardized experimental cohorts:
            
            1. **`sucrose solution 2 microliter`** (Code Name: `Sucrose 2 µL` | Pure 50% w/w Sucrose baseline control, Hazard Ratio $HR \equiv 1.0$)
            2. **`5 % acetone 2 microliter`** (Code Name: `5% Acetone 2 µL`)
            3. **`10 % acetone 2 microliter`** (Code Name: `10% Acetone 2 µL`)
            4. **`5 % acetone 5 microliter`** (Code Name: `5% Acetone 5 µL`)
            
            *Blinded Experiment Mode*: During initial blinded screening when treatment solutions are masked, datasets remain categorized as `None / Blinded` and clean file IDs are displayed.
            """)

        with st.expander("🔄 Rule 3: Resurrected Bee Observation Correction Rule", expanded=True):
            st.markdown("""
            - **Biological Challenge**: In oral toxicology bioassays, bees frequently experience temporary chill-coma, knockdown, or severe lethargy, causing observers to record them as Dead (`D`) on Day $k$. If physiological recovery occurs, the bee may resume normal activity (`A` or `B`) on Day $j > k$.
            - **Algorithmic Correction**: The later alive state takes absolute precedence:
            """)
            st.latex(r"\text{Status}(i, d) = \begin{cases} \text{Alive}, & \text{if } \exists j \ge d \text{ such that observation is } A \text{ or } B \\ \text{Dead}, & \text{otherwise} \end{cases}")
            st.markdown("An individual bee is classified as deceased at day $d^*$ if and only if **all subsequent observations through the end of the experiment remain Dead (`D`)**.")

    # TAB 3: Mathematical Formulations
    with sub_tabs[2]:
        st.markdown("### 📐 Mathematical Formulations & Statistical Methods")
        st.markdown("""
        Every plot generated in Bee Analytics Studio is derived from rigorous, transparent statistical models. Select a model below to review the equation, assumptions, and biological rationale.
        """)

        # 3.1 Logistic
        with st.expander("📈 1. Logistic Sigmoidal Growth Curve", expanded=False):
            st.markdown(r"""
            #### Mathematical Formulation
            Cumulative mortality fraction $P(t) \in [0, 1]$ over elapsed days $t \ge 1.0$ is modeled as:
            """)
            st.latex(r"P(t) = \frac{L}{1 + \exp\left(-k(t - t_0)\right)}")
            st.markdown(r"""
            - **$L \in (0, 1]$**: Upper asymptote denoting maximum cumulative mortality.
            - **$k > 0$**: Intrinsic mortality acceleration rate ($\text{days}^{-1}$).
            - **$t_0$**: Sigmoidal inflection point (day of peak daily mortality rate, where $P(t_0) = L/2$).
            
            **Biological Rationale**: Population tolerance to toxic chemical exposure follows a normal or log-normal distribution; the integral of individual tolerance thresholds manifests as an S-shaped sigmoid curve.
            """)

        # 3.2 Finney Probit & Abbott
        with st.expander("🧪 2. Finney Probit Linear Regression & Abbott's Formula", expanded=False):
            st.markdown(r"""
            #### Abbott's Natural Mortality Correction
            Corrects observed treatment mortality for background control cage mortality:
            """)
            st.latex(r"P_{\text{corrected}} = \frac{P_{\text{treatment}} - P_{\text{control}}}{1 - P_{\text{control}}}")
            st.markdown(r"""
            #### Finney Probit Linearization
            Maps cumulative response probabilities onto standard normal quantiles with a Finney shift of 5:
            """)
            st.latex(r"\text{Probit}(P) = \Phi^{-1}(P) + 5")
            st.latex(r"Y = a + b \cdot \log_{10}(t)")
            st.markdown(r"""
            Fitted using iteratively reweighted least squares with empirical probit weighting:
            """)
            st.latex(r"w_i = \frac{Z_i^2}{P_i (1 - P_i)}, \quad Z_i = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{(Y_i - 5)^2}{2}\right)")
            st.markdown(r"""
            **Goodness-of-Fit $\chi^2$ Test**: $\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$ with degrees of freedom $df = k - 2$. A $p$-value $> 0.05$ indicates excellent linear fit.
            """)

        # 3.3 Lethal Times & Boundary Guards
        with st.expander("⚡ 3. Lethal Time Estimation ($LT_{10} \dots LT_{90}$) & Boundary Guards", expanded=False):
            st.markdown(r"""
            #### Inversion Derivation
            At target lethal fraction $X\%$ ($P = X/100$), compute target probit $Y_X = \Phi^{-1}(P) + 5$:
            - For $LT_{50}$: $Y_{50} = 5.000$
            - For $LT_{10}$: $Y_{10} = 3.718$
            - For $LT_{90}$: $Y_{90} = 6.282$
            
            Inverting the linear equation yields:
            """)
            st.latex(r"\log_{10}(LT_X) = \frac{Y_X - a}{b} \implies LT_X = 10^{\frac{Y_X - a}{b}}")
            st.markdown(r"""
            #### Mathematical Boundary Guards
            In cohorts with negligible mortality (such as untreated sucrose controls), the empirical slope $b \to 0$. Without safeguards, dividing by a near-zero slope produces astronomical numbers (e.g. $24,893\text{ days}$).
            - **Slope Guard**: If $b \le 0.01$, the calculation is guarded and displayed as `Indeterminate` ($\infty$).
            - **Horizon Guard**: Extrapolations beyond $3 \times t_{\max}$ are capped at $> t_{\max}$ with a diagnostic warning.
            """)

        # 3.4 Kaplan-Meier
        with st.expander("📉 4. Kaplan–Meier Survival Estimator & Greenwood Variance", expanded=False):
            st.markdown(r"""
            #### Product-Limit Estimator
            Non-parametric cumulative survival probability at observation day $t$:
            """)
            st.latex(r"\hat{S}(t) = \prod_{t_i \le t} \left(1 - \frac{d_i}{n_i}\right)")
            st.markdown(r"""
            - $t_i$: Days with one or more confirmed deaths.
            - $d_i$: Number of deaths at time $t_i$.
            - $n_i$: Number of living bees at risk immediately prior to $t_i$.
            
            #### Greenwood's Formula for Standard Error
            """)
            st.latex(r"\widehat{\text{Var}}(\hat{S}(t)) = \hat{S}(t)^2 \sum_{t_i \le t} \frac{d_i}{n_i(n_i - d_i)}")
            st.markdown(r"""
            95% log-log confidence intervals ensure estimated bounds remain strictly within $[0, 1]$.
            """)

        # 3.5 Log-Rank & Cumulative Hazard
        with st.expander("📊 5. Mantel–Cox Log-Rank Test & Cumulative Hazard Step Curves", expanded=False):
            st.markdown(r"""
            #### Hypothesis Testing
            Tests the null hypothesis $H_0: S_1(t) = S_2(t) = \dots = S_K(t)$:
            """)
            st.latex(r"E_{ji} = n_{ji} \frac{d_i}{n_i}, \quad \chi^2 = \sum_{j=1}^K \frac{(O_j - E_j)^2}{E_j} \sim \chi^2(K - 1)")
            st.markdown(r"""
            #### Cumulative Hazard Step Curves $H(t)$
            Nelson-Aalen cumulative hazard estimator:
            """)
            st.latex(r"H(t) = -\ln \hat{S}(t) = \sum_{t_i \le t} \frac{d_i}{n_i}")
            st.markdown(r"""
            **Biological Rationale**: A steeper step trajectory indicates a higher instantaneous force of mortality acting on the cohort.
            """)

        # 3.6 Cox Regression & Forest Plot
        with st.expander("🌲 6. Cox Proportional Hazards Regression & Forest Plot", expanded=False):
            st.markdown(r"""
            #### Semi-Parametric Hazard Model
            """)
            st.latex(r"h(t \mid \mathbf{X}) = h_0(t) \exp\left(\sum_{j=1}^p \beta_j X_j\right)")
            st.markdown(r"""
            - $h_0(t)$: Unspecified baseline hazard.
            - **Hazard Ratio ($HR$)**: $HR_j = \exp(\beta_j)$ relative to `Sucrose 2 µL` baseline ($HR \equiv 1.0$).
            - **Forest Plot**: Displays $HR$ points with 95% Wald confidence intervals:
            """)
            st.latex(r"95\% \text{ CI} = \left[\exp\left(\hat{\beta}_j - 1.96 \cdot \text{SE}(\hat{\beta}_j)\right), \, \exp\left(\hat{\beta}_j + 1.96 \cdot \text{SE}(\hat{\beta}_j)\right)\right]")

        # 3.7 Consumption Dynamics
        with st.expander("🍯 7. Sucrose vs. Treatment Solution Consumption Dynamics", expanded=False):
            st.markdown(r"""
            #### Daily Intake Formula
            Daily liquid consumption per living bee ($\mu\text{L} / \text{bee} / \text{day}$):
            """)
            st.latex(r"\text{Consumption}_d = \frac{V_{d-1} - V_d - V_{\text{evaporation}}}{N_{\text{alive}, d}}")
            st.markdown(r"""
            - **Caloric Sucrose Intake (`Cons`)**: Baseline feeding from 50% w/w sucrose solution.
            - **Treatment Solution Intake (`TCons`)**: Ingested volume of test chemical/acetone solution.
            - **Nutritional Anorexia**: Significant drops in sucrose consumption relative to control indicate feeding deterrence, gustatory aversion, or neurotoxic suppression.
            """)

    # TAB 4: Code Snippets
    with sub_tabs[3]:
        st.markdown("### 💻 Reproducible Code Snippets & Pipelines")
        st.markdown("""
        Researchers can run the statistical engine programmatically using standalone Python scripts or pipeline automation.
        """)

        st.code("""
import numpy as np
from modules.core.probit_engine import ProbitEngine
from modules.core.csv_auditor import audit_csv_files

# 1. Audit dataset compliance with Post-2h Starvation Rule
audit = audit_csv_files(["2026-30_export.csv"])
print("Audit status:", audit[0]["status"])

# 2. Run Finney Probit Regression (strictly Day 1 onwards)
days = np.arange(1, 16)
dead_bees = np.array([0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])
total_bees = 10

engine = ProbitEngine(days, dead_bees, total_bees)
fit = engine.fit()

print(f"Slope (b): {fit['slope']:.4f}")
print(f"Intercept (a): {fit['intercept']:.4f}")
print(f"LT50: {fit['LT50']:.2f} Days")
print(f"Goodness-of-Fit Chi2: {fit['chi2']:.3f} (p = {fit['p_value']:.4f})")
        """, language="python")

    # TAB 5: Git Hygiene
    with sub_tabs[4]:
        st.markdown("### 🧹 Repository Hygiene: Untracking Unwanted Files on GitHub")
        st.markdown("""
        #### Why Files Like `.venv_run/` Ended Up on GitHub
        When `git add .` was run in an earlier commit before `.gitignore` was configured, Git staged and committed the entire virtual environment (>10,000 files).
        Adding `.venv_run/` to `.gitignore` **afterwards does NOT untrack already committed files**.

        #### Safe Step-by-Step Fix (Leaves Local Files Untouched)
        Run the following commands in your terminal:
        """)

        st.code("""
# 1. Open your terminal in the repository directory
cd /Users/yakkshit/Downloads/project/hiwi2/p1/new

# 2. Untrack the virtual environment from git index without deleting local files:
git rm -r --cached .venv_run/

# 3. Untrack hidden agent/IDE metadata if tracked:
git rm -r --cached .agents/ .claude/ 2>/dev/null || true

# 4. Commit the cleanup:
git commit -m "chore: purge committed virtual environment from git repository"

# 5. Push to GitHub:
git push origin main
        """, language="bash")

        st.success("✅ **Why this is safe**: The `--cached` flag tells Git to remove the file paths only from the Git index (staging area) and GitHub, without deleting a single file from your computer's hard drive.")

    # TAB 6: Citation
    with sub_tabs[5]:
        st.markdown("### 📝 Citation & Publication Reference")
        st.markdown("""
        If you use Antigravity Bee Analytics Studio in your scientific research, thesis, or publication, please use the following citation:
        """)

        st.code("""
@software{bee_analytics_studio_2026,
  author = {Antigravity Bioassay Consortium},
  title = {Antigravity Bee Analytics Studio: A Publication-Grade Computational Biostatistical Suite for Honeybee (Apis mellifera) Bioassays},
  year = {2026},
  url = {https://github.com/yakkshit/bee}
}
        """, language="bibtex")

if __name__ == "__main__":
    render_doc_page()
