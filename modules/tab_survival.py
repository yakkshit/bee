import pandas as pd
import streamlit as st

from modules.core.time_parser import filter_post_starvation_data
from modules.parser import (
    extract_individual_bee_survival,
    get_default_4_cohort_dataframes,
    parse_batch_df,
    clean_cohort_name,
    get_cohort_code,
    COHORT_PALETTE,
    COHORT_MARKERS
)
from modules.survival_stats import (
    render_kaplan_meier_analysis,
    render_log_rank_analysis,
    render_cox_regression_analysis,
    render_logrank_math_card,
    render_cox_math_card,
    render_kaplan_meier_math_card,
    render_methodology_comparison_card
)




def render_survival_tab():
    """Renders the dedicated Inferential Survival Statistics Laboratory tab."""
    st.header("🔬 Advanced Survival Statistics: Log-Rank Test & Cox Proportional Hazards")
    st.markdown("Perform formal inferential survival statistics: evaluate whether survival distributions differ significantly between exposure groups using the **Log-Rank (Mantel-Cox) Test**, and quantify relative hazard ratios and covariate impacts via **Cox Proportional Hazards Regression**.")

    st.info("🕒 **Bioassay Protocol Active**: Starting observations are strictly post-2h starvation period (Time Zero / baseline d0 is excluded from all kinetics and hazard models).")

    DEFAULT_COLORS = [
        "#2563eb", "#0d9488", "#f59e0b", "#dc2626", "#8b5cf6",
        "#ec4899", "#06b6d4", "#f97316", "#84cc16", "#6366f1"
    ]

    # Automatically take ALL uploaded files (or processed batches / default 4 cohorts)
    uploaded_dict = st.session_state.get('uploaded_file_dict', {})
    batches_avail = st.session_state.get('processed_batches', [])

    all_bees_list = []
    color_map = {}

    if uploaded_dict:
        for idx_u, (fname, (c_name, c_df)) in enumerate(uploaded_dict.items()):
            clean_df = filter_post_starvation_data(c_df.copy())
            b_bees = extract_individual_bee_survival(clean_df, batch_name=c_name)
            if not b_bees.empty:
                b_bees['cohort'] = c_name
                b_bees['cohort_code'] = get_cohort_code(c_name)
                all_bees_list.append(b_bees)
                color_map[c_name] = COHORT_PALETTE.get(c_name, DEFAULT_COLORS[idx_u % len(DEFAULT_COLORS)])
    elif batches_avail:
        for b in batches_avail:
            b_df = b.get('individual_bees')
            if b_df is not None and not b_df.empty:
                b_copy = b_df.copy()
                b_copy['cohort'] = b['name']
                b_copy['cohort_code'] = b.get('code_name', get_cohort_code(b['name']))
                all_bees_list.append(b_copy)
                color_map[b['name']] = b.get('color', '#0284c7')
    else:
        default_dict = get_default_4_cohort_dataframes()
        for c_name, c_df in default_dict.items():
            b_bees = extract_individual_bee_survival(c_df, batch_name=c_name)
            if not b_bees.empty:
                b_bees['cohort'] = c_name
                b_bees['cohort_code'] = get_cohort_code(c_name)
                all_bees_list.append(b_bees)
                color_map[c_name] = COHORT_PALETTE.get(c_name, '#2563eb')

    if not all_bees_list:
        st.info("No cohort data available. Please upload CSV files in the Multi-CSV or Single CSV tab.")
        return

    combined_stat_df = pd.concat(all_bees_list, ignore_index=True)
    all_cohs = list(color_map.keys())

    # Check if single cohort with multiple treatments
    unique_treatments = [t for t in combined_stat_df['treatment'].unique() if pd.notna(t) and str(t).strip() != '']
    if len(all_cohs) == 1 and len(unique_treatments) >= 2:
        group_column = 'treatment'
        analysis_df = combined_stat_df
        st.caption(f"📊 Single dataset with multiple treatments detected: comparing `{', '.join(unique_treatments)}`.")
    elif len(all_cohs) == 1:
        group_column = 'cohort'
        analysis_df = combined_stat_df
    else:
        group_column = 'cohort'
        # BY DEFAULT: ALL UPLOADED FILES ARE TAKEN AND SELECTED!
        sel_cohs = st.multiselect(
            "Select Cohorts to Include in Statistical Comparison:",
            all_cohs,
            default=all_cohs,
            key="tab3_cohort_select",
            help="All uploaded files/cohorts are included by default. Deselect any cohort to exclude it from the comparative tests."
        )
        if len(sel_cohs) == 0:
            st.warning("⚠️ Please select at least one cohort.")
            return
        analysis_df = combined_stat_df[combined_stat_df['cohort'].isin(sel_cohs)].copy()

    if analysis_df is None or analysis_df.empty:
        return


    # Dedicated Subtabs for Kaplan-Meier, Log-Rank, Cox Regression, and Methodology
    subtab_km, subtab_lr, subtab_cox, subtab_math = st.tabs([
        "📉 1. Kaplan–Meier Survival Curves Plot",
        "📊 2. Log-Rank (Mantel-Cox) Test & Observed vs. Expected Plot",
        "🌲 3. Cox Proportional Hazards Regression & Forest Plot",
        "📐 4. Mathematical Methodology & Bioassay Guide"
    ])

    # ==========================================
    # SUBTAB 1: KAPLAN-MEIER SURVIVAL PLOT
    # ==========================================
    with subtab_km:
        st.subheader("1. Kaplan–Meier Multi-Cohort Survival Analysis & Step-by-Step Life Tables")
        col_km_main, col_km_side = st.columns([2.2, 1.0])

        with col_km_main:
            render_kaplan_meier_analysis(
                analysis_df,
                group_col=group_column,
                color_map=color_map,
                key_suffix="tab3_sub_km"
            )

        with col_km_side:
            card_tab_km, card_tab_comp_km = st.tabs(["📐 Kaplan–Meier", "📊 Method Comparison"])
            with card_tab_km:
                render_kaplan_meier_math_card()
            with card_tab_comp_km:
                render_methodology_comparison_card()

    # ==========================================
    # SUBTAB 2: LOG-RANK TEST & OBSERVED VS EXPECTED PLOT
    # ==========================================
    with subtab_lr:
        st.subheader("2. Log-Rank (Mantel-Cox) Hypothesis Test & Observed vs. Expected Mortality")
        col_lr_main, col_lr_side = st.columns([2.2, 1.0])

        with col_lr_main:
            render_log_rank_analysis(
                analysis_df,
                group_col=group_column,
                color_map=color_map,
                key_suffix="tab3_sub_lr"
            )

        with col_lr_side:
            card_tab_lr, card_tab_comp_lr = st.tabs(["📐 Log-Rank", "📊 Method Comparison"])
            with card_tab_lr:
                render_logrank_math_card()
            with card_tab_comp_lr:
                render_methodology_comparison_card()

    # ==========================================
    # SUBTAB 3: COX PROPORTIONAL HAZARDS & FOREST PLOT
    # ==========================================
    with subtab_cox:
        st.subheader("3. Cox Proportional Hazards Model (Hazard Ratios & Forest Plot)")
        col_cox_main, col_cox_side = st.columns([2.2, 1.0])

        with col_cox_main:
            render_cox_regression_analysis(
                analysis_df,
                group_col=group_column,
                key_suffix="tab3_sub_cox"
            )

        with col_cox_side:
            card_tab_cox, card_tab_mc = st.tabs(["🌲 Cox Model", "📊 Method Comparison"])
            with card_tab_cox:
                render_cox_math_card()
            with card_tab_mc:
                render_methodology_comparison_card()

    # ==========================================
    # SUBTAB 4: FULL METHODOLOGICAL GUIDE
    # ==========================================
    with subtab_math:
        st.markdown("### 📚 Comprehensive Bioassay Statistics & Mathematical Framework")
        render_methodology_comparison_card()
        st.markdown("---")
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
