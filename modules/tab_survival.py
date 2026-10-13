import pandas as pd
import streamlit as st

from modules.parser import extract_individual_bee_survival
from modules.survival_stats import render_log_rank_analysis, render_cox_regression_analysis


def render_survival_tab():
    """Renders the dedicated Inferential Survival Statistics Laboratory tab."""
    st.header("🔬 Advanced Survival Statistics: Log-Rank Test & Cox Proportional Hazards")
    st.markdown("Perform formal inferential survival statistics: evaluate whether survival distributions differ significantly between exposure groups using the **Log-Rank (Mantel-Cox) Test**, and quantify relative hazard ratios and covariate impacts via **Cox Proportional Hazards Regression**.")

    stat_source = st.radio(
        "Select Dataset Source for Survival Statistics:",
        ["📂 Multi-CSV Comparison Cohorts", "📄 Single CSV Dataset"],
        horizontal=True
    )

    if stat_source.startswith("📂 Multi-CSV"):
        batches_avail = st.session_state.get('processed_batches', [])

        if batches_avail:
            all_bees_list = []
            c_map_tab3 = {}
            for b in batches_avail:
                b_df = b.get('individual_bees')
                if b_df is not None and not b_df.empty:
                    b_copy = b_df.copy()
                    b_copy['cohort'] = b['name']
                    all_bees_list.append(b_copy)
                    c_map_tab3[b['name']] = b.get('color', '#0284c7')

            if len(all_bees_list) >= 2:
                combined_stat_df = pd.concat(all_bees_list, ignore_index=True)
                all_cohs = list(c_map_tab3.keys())
                sel_cohs = st.multiselect(
                    "Select Cohorts to Include in Statistical Comparison:",
                    all_cohs, default=all_cohs, key="tab3_cohort_select"
                )

                if len(sel_cohs) >= 2:
                    filtered_df = combined_stat_df[combined_stat_df['cohort'].isin(sel_cohs)].copy()

                    st.markdown("---")
                    st.subheader("1. Log-Rank (Mantel-Cox) Multi-Cohort Survival Test")
                    render_log_rank_analysis(filtered_df, group_col='cohort', color_map=c_map_tab3)

                    st.markdown("---")
                    st.subheader("2. Cox Proportional Hazards Model (Cohort Hazard Ratios & Covariates)")
                    render_cox_regression_analysis(filtered_df, group_col='cohort')
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

            if len(treatment_vals) >= 2:
                st.markdown("---")
                st.subheader("1. Log-Rank (Mantel-Cox) Survival Test (By Treatment)")
                render_log_rank_analysis(single_bees_df, group_col='treatment')
            else:
                st.info("💡 **Log-Rank Note**: The single CSV file contains only 1 treatment value. For multi-cohort comparison, upload multiple CSV files in Tab 2 or include distinct labels in the 'Treatment' column.")

            st.markdown("---")
            st.subheader("2. Cox Proportional Hazards Regression (Individual Bee Covariates)")
            render_cox_regression_analysis(single_bees_df, group_col='treatment' if len(treatment_vals) > 1 else 'cohort')
        else:
            st.info("No single CSV data loaded yet. Please view the '🐝 Single CSV Analysis' tab.")
