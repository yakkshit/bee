"""
Global Plot Search & Interactive Navigation Engine.
Allows researchers to instantly search, locate, preview, and download any plot
or statistical distribution across Single-CSV and Multi-CSV comparative modules.
"""
import streamlit as st


PLOT_REGISTRY = [
    {
        "id": "probit_multi",
        "name": "Comparative Probit Regression Across CSV Files",
        "tab": "Multi-CSV Probit Comparison",
        "section": "Section 1: Comparative Probit Regression",
        "keywords": ["probit", "regression", "lt50", "lt20", "lethal time", "slope", "dose-response", "confidence ribbon"],
        "category": "Probit & Mortality",
        "filename": "probit_comparison.png",
        "description": "Fitted Probit regression lines (Y = a + bx) across all cohorts with 95% CI shaded bands, slope standard error SE(b), and residual degrees of freedom."
    },
    {
        "id": "mortality_multi",
        "name": "Abbott-Corrected Mortality Curve Comparison",
        "tab": "Multi-CSV Probit Comparison",
        "section": "Section 2: Mortality Curve Comparison",
        "keywords": ["mortality", "abbott", "death", "percentage", "confidence interval", "whiskers", "alternative"],
        "category": "Probit & Mortality",
        "filename": "mortality_comparison.png",
        "description": "Trajectory of Abbott-corrected cumulative mortality over observation days. Available in standard view (with SEM whiskers) and clean alternative view."
    },
    {
        "id": "survival_multi",
        "name": "Cohort Survival Rate Curve Comparison",
        "tab": "Multi-CSV Probit Comparison",
        "section": "Section 3: Survival Curve Comparison",
        "keywords": ["survival", "rate", "percentage", "lifespan", "whiskers", "alternative"],
        "category": "Survival & Lifespan",
        "filename": "survival_comparison.png",
        "description": "Direct percentage survival curve comparison over time with optional SEM/SD error whiskers and clean alternative view."
    },
    {
        "id": "kaplan_meier",
        "name": "Kaplan–Meier Survival Step Curves",
        "tab": "Multi-CSV Probit Comparison / Survival Statistics",
        "section": "Section 6 > Sub-tab 1: Kaplan–Meier Survival Curves",
        "keywords": ["kaplan-meier", "km", "product-limit", "step curve", "greenwood", "censored", "survival probability"],
        "category": "Survival & Lifespan",
        "filename": "kaplan_meier_survival.png",
        "description": "Non-parametric product-limit Kaplan-Meier estimator with Greenwood's formula 95% confidence bands and tick markers for right-censored bees."
    },
    {
        "id": "log_rank_hazard",
        "name": "Log-Rank (Mantel-Cox) Cumulative Hazard H(t)",
        "tab": "Multi-CSV Probit Comparison / Survival Statistics",
        "section": "Section 6 > Sub-tab 2: Log-Rank (Mantel-Cox) Test",
        "keywords": ["log-rank", "mantel-cox", "cumulative hazard", "chi-square", "p-value", "hazard function"],
        "category": "Survival & Lifespan",
        "filename": "logrank_cumulative_hazard.png",
        "description": "Nelson-Aalen / Log-Rank cumulative hazard trajectories across experimental cohorts."
    },
    {
        "id": "log_rank_oe",
        "name": "Mantel-Cox Observed vs. Expected Deaths (O vs. E)",
        "tab": "Multi-CSV Probit Comparison / Survival Statistics",
        "section": "Section 6 > Sub-tab 2: Log-Rank (Mantel-Cox) Test",
        "keywords": ["observed vs expected", "o vs e", "excess deaths", "wald z", "relative risk", "bar chart"],
        "category": "Survival & Lifespan",
        "filename": "logrank_observed_vs_expected.png",
        "description": "Comparative bar chart of observed deaths against null-expected deaths under H0 across cohorts."
    },
    {
        "id": "lifespan_histogram",
        "name": "Cross-Cohort Lifespan Distribution Histogram (with Hypothesis Test p-values)",
        "tab": "Multi-CSV Probit Comparison / Survival Statistics",
        "section": "Section 6 > Sub-tab 2 & Sub-tab 4",
        "keywords": ["histogram", "lifespan", "distribution", "p-value", "kruskal-wallis", "log-rank", "frequency", "days"],
        "category": "Histograms & P-Values",
        "filename": "logrank_lifespan_histogram.png",
        "description": "Overlaid lifespan frequency histograms across cohorts reporting both Mantel-Cox Log-Rank p-value and Kruskal-Wallis rank test p-value."
    },
    {
        "id": "multi_testing_pvalues",
        "name": "Multi-Testing Correction & FDR Comparison Plot (-log10 p)",
        "tab": "Multi-CSV Probit Comparison",
        "section": "Section 6 > Sub-tab 4: Multi-Testing & FDR Corrections",
        "keywords": ["multiple testing", "fdr", "bonferroni", "holm", "benjamini-hochberg", "bh", "by", "sidak", "p-value", "q-value"],
        "category": "Multiple Testing & FDR",
        "filename": "multiple_testing_correction_pvalues.png",
        "description": "Stem scatter plot comparing Raw p-values against Bonferroni, Holm, Benjamini-Hochberg (BH FDR), Benjamini-Yekutieli (BY FDR), and Šidák corrections with alpha=0.05 cutoff."
    },
    {
        "id": "pvalue_histogram",
        "name": "Pairwise Hypothesis Test P-Value Distribution Histogram",
        "tab": "Multi-CSV Probit Comparison",
        "section": "Section 6 > Sub-tab 4: Multi-Testing & FDR Corrections",
        "keywords": ["p-value histogram", "p-value distribution", "uniform", "null", "spike", "false discovery", "fdr", "histogram"],
        "category": "Histograms & P-Values",
        "filename": "pairwise_pvalue_distribution_histogram.png",
        "description": "Frequency distribution of empirical pairwise p-values evaluated against uniform null expectation U(0,1) with alpha=0.05 significance cutoff."
    },
    {
        "id": "cox_forest",
        "name": "Cox Proportional Hazards Regression & Forest Plot",
        "tab": "Multi-CSV Probit Comparison / Survival Statistics",
        "section": "Section 6 > Sub-tab 3: Cox Proportional Hazards",
        "keywords": ["cox", "forest plot", "hazard ratio", "hr", "confidence interval", "relative risk", "wald"],
        "category": "Survival & Lifespan",
        "filename": "cox_forest_plot.png",
        "description": "Forest plot depicting multivariate Hazard Ratios (HR) with 95% Confidence Intervals against the baseline control cohort."
    },
    {
        "id": "grouped_bar_cons",
        "name": "Day-by-Day Sucrose Consumption Grouped Bar Chart",
        "tab": "Multi-CSV Probit Comparison",
        "section": "Section 5.2: Day-by-Day Grouped Bar Chart Comparison",
        "keywords": ["consumption", "sucrose", "intake", "grouped bar", "error bar", "whisker", "sem", "sd", "day-by-day"],
        "category": "Consumption & Feeding",
        "filename": "day_by_day_grouped_bar_consumption.png",
        "description": "Day-by-day comparative grouped bar chart of mean sucrose intake (µL/bee) across all cohorts, with black error whiskers denoting ±1 SEM or ±1 SD."
    },
    {
        "id": "grouped_bar_tcons",
        "name": "Day-by-Day Treatment Solution Intake Grouped Bar Chart",
        "tab": "Multi-CSV Probit Comparison",
        "section": "Section 5.2: Day-by-Day Grouped Bar Chart Comparison",
        "keywords": ["treatment consumption", "tcons", "intake", "grouped bar", "whisker", "sem", "sd"],
        "category": "Consumption & Feeding",
        "filename": "day_by_day_grouped_bar_treatment_consumption.png",
        "description": "Day-by-day grouped bar comparison for active treatment solution intake (TCons) across experimental cohorts."
    },
    {
        "id": "daily_status_dist",
        "name": "Daily Bee Status Distribution Histogram (Alive vs Dead with Chi² Test)",
        "tab": "Single CSV Analysis",
        "section": "Section 4: Feeding & Behavioral Kinetics > Daily Status Distribution",
        "keywords": ["single csv", "status distribution", "histogram", "alive", "dead", "chi-square", "goodness-of-fit", "daily"],
        "category": "Histograms & P-Values",
        "filename": "06_daily_status_distribution_histogram.png",
        "description": "Stacked daily histogram of Alive vs Dead bee counts with automated Goodness-of-Fit Chi-Square test p-value."
    },
    {
        "id": "bee_lifespan_dist",
        "name": "Individual Bee Lifespan Distribution Histogram (with Shapiro-Wilk Normality p-value)",
        "tab": "Single CSV Analysis",
        "section": "Section 4: Feeding & Behavioral Kinetics > Lifespan Distribution",
        "keywords": ["single csv", "lifespan", "shapiro-wilk", "normality", "p-value", "histogram", "skewness"],
        "category": "Histograms & P-Values",
        "filename": "07_bee_lifespan_distribution_histogram.png",
        "description": "Frequency histogram of individual bee survival times within a single cohort with Shapiro-Wilk normality test p-value and mean/median markers."
    },
    {
        "id": "daily_sucrose_single",
        "name": "Daily Sucrose Consumption Trajectory (Single Cohort)",
        "tab": "Single CSV Analysis",
        "section": "Section 4: Feeding & Behavioral Kinetics > Daily Intake Trajectory",
        "keywords": ["single csv", "sucrose consumption", "sem ribbon", "trajectory", "intake", "daily mean"],
        "category": "Consumption & Feeding",
        "filename": "08_daily_sucrose_consumption.png",
        "description": "Time trajectory of mean daily sucrose consumption per bee with shaded ±1 SEM precision ribbons."
    },
    {
        "id": "feeding_persistence",
        "name": "Feeding Cessation / Anorexia Kinetics (Kaplan-Meier)",
        "tab": "Single CSV Analysis / Multi-CSV",
        "section": "Section 4 / Section 5.3: Feeding Cessation Step Curves",
        "keywords": ["feeding cessation", "anorexia", "persistence", "kaplan-meier", "step curve"],
        "category": "Consumption & Feeding",
        "filename": "09_feeding_persistence_km.png",
        "description": "Time-to-anorexia survival kinetics tracking when bees cease active feeding prior to mortality."
    }
]


def render_global_plot_search():
    """Renders an interactive search bar & plot navigator at the top of the studio."""
    with st.expander("🔍 **Global Plot Search & Direct Navigator** (Find any plot, histogram, or analysis immediately)", expanded=False):
        c_srch, c_cat = st.columns([2.5, 1.5])
        with c_srch:
            query = st.text_input(
                "Search all plots by name, statistical test, or keyword:",
                placeholder="e.g. probit, bonferroni, fdr, histogram, p-value, consumption, whiskers, log-rank...",
                key="global_plot_search_query"
            ).strip().lower()

        categories = ["All Categories", "Probit & Mortality", "Survival & Lifespan", "Multiple Testing & FDR", "Histograms & P-Values", "Consumption & Feeding"]
        with c_cat:
            cat_choice = st.selectbox("Filter by Category:", categories, index=0, key="global_plot_search_cat")

        # Filter matches
        matches = []
        for item in PLOT_REGISTRY:
            if cat_choice != "All Categories" and item["category"] != cat_choice:
                continue
            if not query:
                matches.append(item)
            else:
                q_words = query.split()
                text_to_search = f"{item['name']} {item['tab']} {item['section']} {item['description']} {' '.join(item['keywords'])}".lower()
                if all(w in text_to_search for w in q_words):
                    matches.append(item)

        st.caption(f"Showing **{len(matches)}** matching plot(s) out of {len(PLOT_REGISTRY)} registered analyses:")

        if not matches:
            st.warning(f"No plots matched your search query '{query}'. Try searching for 'probit', 'fdr', 'survival', 'histogram', or 'consumption'.")
            return

        # Display matching plots in clean cards with direct preview/download if cached
        plot_cache = st.session_state.get('multi_plot_images', {})

        for m_item in matches:
            with st.container():
                st.markdown(f"""
                <div style="background: white; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px; border-left: 4px solid #0284c7; box-shadow: 0 1px 3px rgba(0,0,0,0.06);">
                    <div style="display: flex; justify-content: space-between; align-items: baseline;">
                        <span style="font-size: 1.05rem; font-weight: bold; color: #0f172a;">{m_item['name']}</span>
                        <span style="background: #e0f2fe; color: #0369a1; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: bold;">{m_item['category']}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #64748b; margin-top: 3px;">
                        📍 <b>Location:</b> {m_item['tab']} &nbsp;›&nbsp; {m_item['section']}
                    </div>
                    <div style="font-size: 0.88rem; color: #334155; margin-top: 6px;">
                        {m_item['description']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Check if generated image exists in session_state
                fn = m_item['filename']
                img_bytes = plot_cache.get(fn)

                c_info1, c_info2 = st.columns([3, 1])
                with c_info2:
                    if img_bytes:
                        st.download_button(
                            label=f"📥 Download PNG",
                            data=img_bytes,
                            file_name=fn,
                            mime="image/png",
                            key=f"dl_search_{m_item['id']}",
                            use_container_width=True
                        )
                    else:
                        st.caption("ℹ️ *Generates when data is loaded*")

                with c_info1:
                    if img_bytes:
                        with st.popover(f"👁️ Quick Preview: {m_item['name']}"):
                            try:
                                st.image(img_bytes, caption=m_item['name'], use_column_width=True)
                            except TypeError:
                                st.image(img_bytes, caption=m_item['name'])

