import pandas as pd
import streamlit as st

from modules.parser import extract_individual_bee_survival
from modules.survival_stats import (
    render_log_rank_analysis,
    render_cox_regression_analysis,
    render_logrank_math_card,
    render_cox_math_card
)


def render_survival_tab():
    """Renders the dedicated Inferential Survival Statistics Laboratory tab."""
    st.header("🔬 Advanced Survival Statistics: Log-Rank Test & Cox Proportional Hazards")
    st.markdown("Perform formal inferential survival statistics: evaluate whether survival distributions differ significantly between exposure groups using the **Log-Rank (Mantel-Cox) Test**, and quantify relative hazard ratios and covariate impacts via **Cox Proportional Hazards Regression**.")

    stat_source = st.radio(
        "Select Dataset Source for Survival Statistics:",
        ["📂 Multi-CSV Comparison Cohorts", "📄 Single CSV Dataset"],
        horizontal=True
    )

    # Prepare dataset
    analysis_df = None
    group_column = 'cohort'
    color_map = {}
    is_valid = False

    if stat_source.startswith("📂 Multi-CSV"):
        batches_avail = st.session_state.get('processed_batches', [])
        if batches_avail:
            all_bees_list = []
            for b in batches_avail:
                b_df = b.get('individual_bees')
                if b_df is not None and not b_df.empty:
                    b_copy = b_df.copy()
                    b_copy['cohort'] = b['name']
                    all_bees_list.append(b_copy)
                    color_map[b['name']] = b.get('color', '#0284c7')

            if len(all_bees_list) >= 2:
                combined_stat_df = pd.concat(all_bees_list, ignore_index=True)
                all_cohs = list(color_map.keys())
                sel_cohs = st.multiselect(
                    "Select Cohorts to Include in Statistical Comparison:",
                    all_cohs, default=all_cohs, key="tab3_cohort_select"
                )
                if len(sel_cohs) >= 2:
                    analysis_df = combined_stat_df[combined_stat_df['cohort'].isin(sel_cohs)].copy()
                    group_column = 'cohort'
                    is_valid = True
                else:
                    st.info("Please select at least 2 cohorts to compare.")
            else:
                st.info("At least 2 cohorts are required to perform survival comparisons. Please upload CSV files in the Multi-CSV tab.")
        else:
            st.info("No Multi-CSV cohorts found yet. Please view the '📈 Multi-CSV Probit Comparison' tab or upload your files to initialize.")

    else:
        s_df = st.session_state.get('df')
        if s_df is not None:
            single_bees_df = extract_individual_bee_survival(s_df, batch_name="Current Dataset")
            st.markdown(f"**Loaded Dataset:** {len(single_bees_df)} individual bees observed over {len(s_df)} days.")
            treatment_vals = [str(x).strip() for x in s_df['Treatment'].unique() if pd.notna(x) and str(x).strip() != ''] if 'Treatment' in s_df.columns else []

            analysis_df = single_bees_df
            group_column = 'treatment' if len(treatment_vals) >= 2 else 'cohort'
            is_valid = True
        else:
            st.info("No single CSV data loaded yet. Please view the '🐝 Single CSV Analysis' tab.")

    if not is_valid or analysis_df is None:
        return

    # Dedicated Subtabs for Cox Regression, Log-Rank, and Mathematical Framework
    subtab_lr, subtab_cox, subtab_math = st.tabs([
        "📉 Log-Rank (Mantel-Cox) Comparison",
        "🌲 Cox Proportional Hazards Regression",
        "📐 Mathematical Methodology & Bioassay Guide"
    ])

    # ==========================================
    # SUBTAB 1: LOG-RANK TEST
    # ==========================================
    with subtab_lr:
        st.subheader("1. Log-Rank (Mantel-Cox) Multi-Cohort Survival Test")
        col_lr_main, col_lr_side = st.columns([2.2, 1.0])

        with col_lr_main:
            render_log_rank_analysis(
                analysis_df,
                group_col=group_column,
                color_map=color_map,
                key_suffix="tab3_sub_lr"
            )

        with col_lr_side:
            render_logrank_math_card()

    # ==========================================
    # SUBTAB 2: COX PROPORTIONAL HAZARDS
    # ==========================================
    with subtab_cox:
        st.subheader("2. Cox Proportional Hazards Model (Hazard Ratios & Uncertainty)")
        col_cox_main, col_cox_side = st.columns([2.2, 1.0])

        with col_cox_main:
            render_cox_regression_analysis(
                analysis_df,
                group_col=group_column,
                key_suffix="tab3_sub_cox"
            )

        with col_cox_side:
            render_cox_math_card()

    # ==========================================
    # SUBTAB 3: FULL METHODOLOGICAL GUIDE
    # ==========================================
    with subtab_math:
        st.markdown("### 📚 Comprehensive Bioassay Statistics & Mathematical Framework")
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown("""
            #### 1. Finney Probit Regression & Lethal Times ($LT_X$)
            - **Probit Transformation:**
              $$\\text{Probit}(P) = \\Phi^{-1}(P) + 5.0$$
            - **Linear Regression Line:**
              $$\\text{Probit}(P) = a + b \\cdot \\log_{10}(t)$$
            - **Lethal Time for Percentile $X\\%$ ($LT_X$):**
              $$LT_X = 10^{\\frac{\\Phi^{-1}(X/100) + 5.0 - a}{b}}$$
            - **Abbott's Natural Control Mortality Correction:**
              $$P_{\\text{corrected}} = \\frac{P_{\\text{observed}} - C}{100 - C} \\times 100\\%$$

            #### 2. Kaplan-Meier Product-Limit Estimator
            - **Non-Parametric Survival Function:**
              $$\\hat{S}(t) = \\prod_{t_i \\le t} \\left(1 - \\frac{d_i}{n_i}\\right)$$
            - **Greenwood's Standard Error (95% Confidence Band):**
              $$\\widehat{\\text{SE}}(\\hat{S}(t)) = \\hat{S}(t) \\sqrt{\\sum_{t_i \\le t} \\frac{d_i}{n_i (n_i - d_i)}}$$
            """)

        with col_m2:
            st.markdown("""
            #### 3. Log-Rank (Mantel-Cox) Rank Test
            - **Mantel-Haenszel Chi-Square Statistic:**
              $$\\chi^2 = \\frac{\\left(\\sum_{j=1}^K (O_{1j} - E_{1j})\\right)^2}{\\sum_{j=1}^K V_{1j}}$$
            - **Hypothesis Testing:**
              - $H_0$: Equal survival distributions across cohorts ($S_1(t) = S_2(t) = \\dots = S_k(t)$).
              - $H_1$: At least one cohort exhibits significantly different hazard.
              - Post-hoc pairwise tests utilize **Bonferroni multiplicity correction** $\\alpha_{\\text{adj}} = \\alpha / k$.

            #### 4. Cox Proportional Hazards Model & Uncertainty
            - **Semi-Parametric Hazard Equation:**
              $$h(t \\mid X) = h_0(t) \\exp(\\beta_1 X_1 + \\dots + \\beta_p X_p)$$
            - **Hazard Ratio Point Estimate:**
              $$\\text{HR} = \\exp(\\beta)$$
            - **95% Confidence Interval (Parameter Uncertainty):**
              $$\\text{CI}_{95\\%} = \\left[\\exp(\\beta - 1.96 \\cdot \\text{SE}), \\; \\exp(\\beta + 1.96 \\cdot \\text{SE})\\right]$$
            - An interval excluding $1.0$ indicates statistical significance ($p < 0.05$).
            """)
