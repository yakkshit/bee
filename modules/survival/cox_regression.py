"""
Cox Proportional Hazards Regression Analysis Module.
Fits semi-parametric survival models, quantifies covariate hazard effects,
and manages interpretation and reporting.
"""
import numpy as np
import pandas as pd
import streamlit as st
from lifelines import CoxPHFitter

from modules.parser import clean_cohort_name, get_cohort_code, COHORT_SUCROSE_2UL
from modules.survival.forest_plot import render_hazard_ratio_forest_plot


def render_cox_regression_analysis(survival_df, group_col='cohort', reference_group=None, key_suffix: str = ""):
    """
    Fits Cox Proportional Hazards Model, computes Hazard Ratios, C-index,
    and generates a Forest Plot with error whiskers.
    """
    if survival_df is None or survival_df.empty:
        st.warning("No individual survival data available for Cox Regression.")
        return

    raw_groups = [g for g in survival_df[group_col].unique() if pd.notna(g) and str(g).strip() != ''] if (group_col and group_col in survival_df.columns) else []

    # Sort groups so baseline reference control (sucrose) is prioritized
    def cohort_priority(g):
        c_clean = clean_cohort_name(g)
        if c_clean == COHORT_SUCROSE_2UL or any(k in str(g).lower() for k in ["sucrose", "control", "ctrl", "water", "0um"]):
            return 0
        elif "5 % acetone 2" in c_clean or "5% acetone 2" in str(g).lower():
            return 1
        elif "10 % acetone 2" in c_clean or "10% acetone 2" in str(g).lower():
            return 2
        elif "5 % acetone 5" in c_clean or "5% acetone 5" in str(g).lower():
            return 3
        elif "batch a" in str(g).lower():
            return -1
        return 10

    groups = sorted(raw_groups, key=cohort_priority)

    st.markdown("Configure covariates for the Cox Proportional Hazards Model:")
    c_cfg1, c_cfg2 = st.columns([1, 2])
    with c_cfg1:
        if len(groups) > 1:
            ref_k = f"cox_ref_{group_col}_{key_suffix}" if key_suffix else f"cox_ref_{group_col}"
            # Default to control / sucrose group
            default_ref_idx = 0
            for idx_g, g_val in enumerate(groups):
                if cohort_priority(g_val) <= 0:
                    default_ref_idx = idx_g
                    break
            ref_selected = st.selectbox("Baseline Reference Cohort (HR = 1.0)", groups, index=default_ref_idx, key=ref_k)
        else:
            ref_selected = groups[0] if groups else None

    with c_cfg2:
        cov_options = []
        for col_key, col_label in [('mean_consumption', "Mean Consumption (µL/day)"), ('temp', "Temperature (°C)"), ('humidity', "Humidity (%)")]:
            if col_key in survival_df.columns:
                s_clean = pd.to_numeric(survival_df[col_key], errors='coerce').dropna()
                if len(s_clean) > 1 and s_clean.nunique() > 1 and float(s_clean.std()) > 0.05:
                    cov_options.append(col_label)

        cov_k = f"cox_covs_{group_col}_{key_suffix}" if key_suffix else f"cox_covs_{group_col}"
        selected_covs = st.multiselect("Additional Covariates to Include in Cox Model:", cov_options, default=cov_options, key=cov_k)

    cph_df = survival_df[['duration', 'event']].copy()

    if len(groups) > 1 and ref_selected is not None:
        dummies = pd.get_dummies(survival_df[group_col], prefix='Cohort', dtype=float)
        ref_col = f"Cohort_{ref_selected}"
        if ref_col in dummies.columns:
            dummies = dummies.drop(columns=[ref_col])
        valid_dummies = [c for c in dummies.columns if dummies[c].nunique() > 1 and dummies[c].std() > 0.01]
        if valid_dummies:
            cph_df = pd.concat([cph_df, dummies[valid_dummies]], axis=1)

    if "Mean Consumption (µL/day)" in selected_covs and 'mean_consumption' in survival_df.columns:
        s_c = pd.to_numeric(survival_df['mean_consumption'], errors='coerce').fillna(0.0)
        if s_c.nunique() > 1 and s_c.std() > 0.05:
            cph_df['Mean_Consumption_uL'] = s_c.astype(float)

    if "Temperature (°C)" in selected_covs and 'temp' in survival_df.columns:
        s_t = pd.to_numeric(survival_df['temp'], errors='coerce').fillna(25.0)
        if s_t.nunique() > 1 and s_t.std() > 0.05:
            cph_df['Temp_C'] = s_t.astype(float)

    if "Humidity (%)" in selected_covs and 'humidity' in survival_df.columns:
        s_h = pd.to_numeric(survival_df['humidity'], errors='coerce').fillna(60.0)
        if s_h.nunique() > 1 and s_h.std() > 0.05:
            cph_df['Humidity_pct'] = s_h.astype(float)

    covariate_cols = [c for c in cph_df.columns if c not in ['duration', 'event']]
    if not covariate_cols:
        st.info("💡 **Cox Proportional Hazards Model**: Requires at least 2 distinct cohorts (or individual covariates with variance such as feeding consumption) to quantify hazard ratios. Please select additional cohorts or covariates above.")
        return

    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(cph_df, duration_col='duration', event_col='event')
    except Exception as ex:
        st.error(f"Error fitting Cox Proportional Hazards Model: {ex}")
        return

    c_index = cph.concordance_index_
    lrt = cph.log_likelihood_ratio_test()
    aic_val = cph.AIC_partial_

    ck1, ck2, ck3, ck4 = st.columns(4)
    with ck1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{c_index:.3f}</div><div class="kpi-label">Concordance Index (C-index)</div></div>', unsafe_allow_html=True)
    with ck2:
        lrt_p_str = "< 0.0001" if lrt.p_value < 0.0001 else f"{lrt.p_value:.4f}"
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{lrt.test_statistic:.2f} (p={lrt_p_str})</div><div class="kpi-label">Likelihood Ratio Test (χ²)</div></div>', unsafe_allow_html=True)
    with ck3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{aic_val:.2f}</div><div class="kpi-label">Partial AIC</div></div>', unsafe_allow_html=True)
    with ck4:
        c_qual = "Strong (>0.80)" if c_index >= 0.8 else ("Good (>0.70)" if c_index >= 0.7 else "Moderate")
        st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color: #0284c7;">{c_qual}</div><div class="kpi-label">Discrimination Power</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    s = cph.summary
    table_rows = []

    # 1. Insert Baseline Reference Cohort row so all cohorts are fully represented
    if len(groups) > 1 and ref_selected is not None:
        ref_clean = clean_cohort_name(ref_selected)
        ref_code = get_cohort_code(ref_selected)
        table_rows.append({
            "Agreed Cohort": ref_clean,
            "Code Name": ref_code,
            "Covariate": f"{ref_code} (Reference Baseline)",
            "Coef (β)": 0.0,
            "Hazard Ratio (HR)": 1.0,
            "95% CI Lower": 1.0,
            "95% CI Upper": 1.0,
            "95% CI Uncertainty": "[1.00 – 1.00]",
            "CI Margin (±)": 0.0,
            "Std Error (SE)": 0.0,
            "Wald z": 0.0,
            "p-value": "1.00 (Ref)",
            "Significance": "⚪ Reference Baseline",
            "is_reference": True
        })

    for var_name, row in s.iterrows():
        b_coef = float(row['coef'])
        se_val = float(row['se(coef)']) if pd.notna(row['se(coef)']) else 0.0

        raw_ci_l = float(row['coef lower 95%']) if 'coef lower 95%' in row and pd.notna(row['coef lower 95%']) else (b_coef - 1.96 * se_val)
        raw_ci_u = float(row['coef upper 95%']) if 'coef upper 95%' in row and pd.notna(row['coef upper 95%']) else (b_coef + 1.96 * se_val)

        ci_l = float(np.exp(np.clip(raw_ci_l, -25.0, 25.0)))
        ci_u = float(np.exp(np.clip(raw_ci_u, -25.0, 25.0)))
        hr_val = float(np.exp(np.clip(b_coef, -25.0, 25.0)))

        if not np.isfinite(hr_val) or hr_val <= 0:
            hr_val = 1.0
        if not np.isfinite(ci_l) or ci_l <= 0:
            ci_l = 0.01
        if not np.isfinite(ci_u) or ci_u <= 0:
            ci_u = max(ci_l * 1.5, hr_val * 2.0, 2.0)
        if ci_u < ci_l:
            ci_u = ci_l * 1.5

        z_v = float(row['z']) if pd.notna(row['z']) else 0.0
        p_v = float(row['p']) if pd.notna(row['p']) else 1.0
        sig_str = "✅ p < 0.05" if p_v < 0.05 else "⚪ n.s."

        if str(var_name).startswith("Cohort_"):
            raw_c_name = str(var_name)[len("Cohort_"):]
            full_c = clean_cohort_name(raw_c_name)
            code_c = get_cohort_code(raw_c_name)
            display_name = code_c
        else:
            full_c = str(var_name).replace("_", " ")
            code_c = full_c
            display_name = full_c

        ci_margin = round((ci_u - ci_l) / 2.0, 3)
        table_rows.append({
            "Agreed Cohort": full_c,
            "Code Name": code_c,
            "Covariate": display_name,
            "Coef (β)": round(b_coef, 4),
            "Hazard Ratio (HR)": round(hr_val, 4),
            "95% CI Lower": round(ci_l, 4),
            "95% CI Upper": round(ci_u, 4),
            "95% CI Uncertainty": f"[{ci_l:.2f} – {ci_u:.2f}]",
            "CI Margin (±)": ci_margin,
            "Std Error (SE)": round(se_val, 4),
            "Wald z": round(z_v, 3),
            "p-value": f"{p_v:.4g}",
            "Significance": sig_str,
            "is_reference": False
        })

    res_df = pd.DataFrame(table_rows)
    st.markdown("#### 📋 Cox Proportional Hazards Model Coefficients & Hazard Ratios")
    if ref_selected is not None and len(groups) > 1:
        ref_code_str = get_cohort_code(ref_selected)
        ref_clean_str = clean_cohort_name(ref_selected)
        st.caption(f"📌 **Baseline Reference Cohort**: `{ref_clean_str}` (`{ref_code_str}`) with reference hazard ratio $HR = 1.00$. All cohort hazard ratios below quantify relative instantaneous mortality risk against this baseline.")
    st.dataframe(res_df, use_container_width=True)

    # Render Forest Plot
    render_hazard_ratio_forest_plot(table_rows, group_col=group_col, key_suffix=key_suffix, res_df=res_df)

    # Biological Hazard & Uncertainty Interpretation
    st.markdown("#### 💡 Biological Hazard & Uncertainty Interpretation")
    for r in table_rows:
        if r.get("is_reference", False):
            st.write(f"- **{r['Agreed Cohort']} ({r['Code Name']})**: Baseline reference group ($HR = 1.00$, Null Hazard Reference).")
            continue

        hr_item = r["Hazard Ratio (HR)"]
        ci_str = r["95% CI Uncertainty"]
        p_val_str = str(r["p-value"]).split()[0]
        try:
            p_item = float(p_val_str)
        except Exception:
            p_item = 1.0
        cov_name = r["Covariate"]
        if r.get('Agreed Cohort') and r.get('Code Name') and r.get('Agreed Cohort') != r.get('Code Name'):
            cohort_display = f"{r['Agreed Cohort']} ({r['Code Name']})"
        else:
            cohort_display = r.get('Code Name') or r.get('Agreed Cohort') or cov_name

        if p_item < 0.05:
            if hr_item > 1.0:
                pct_inc = (hr_item - 1.0) * 100.0
                st.write(f"- **{cohort_display}**: Associated with a **{pct_inc:.1f}% increase** in instantaneous mortality risk ($HR = {hr_item:.2f}$, 95% CI: `{ci_str}`, $p = {p_item:.4g}$). Accelerates mortality with statistical significance.")
            else:
                pct_dec = (1.0 - hr_item) * 100.0
                st.write(f"- **{cohort_display}**: Associated with a **{pct_dec:.1f}% decrease** in instantaneous mortality risk ($HR = {hr_item:.2f}$, 95% CI: `{ci_str}`, $p = {p_item:.4g}$). Conveys relative survival protection.")
        else:
            st.write(f"- **{cohort_display}**: No statistically significant hazard effect detected ($HR = {hr_item:.2f}$, 95% CI: `{ci_str}`, $p = {p_item:.4g}$). The confidence interval crosses 1.0, reflecting uncertainty consistent with null effect.")

