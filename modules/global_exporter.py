"""
Global Exporter Module for Bee Analytics Studio.
Provides unified, one-click global download capabilities across the entire application:
- For every uploaded file: complete single-file plot bundle (Probit, Abbott mortality, Kaplan-Meier, Publication 4-panel, Lethal time profile, and CSV data)
- Multi-cohort comparison plots (Comparative Probit, Faceted Grid, Mortality, Survival, LT50 bars, and Summary CSVs)
- Comprehensive Studio Master Bundle (All Multi plots + All Single plots for each file)
Renderable in both the header quick-actions bar, the sidebar, and Tab 2.
"""
import io
import re
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt

from modules.single_batch.computation import compute_single_batch_data
from modules.core.cohort_manager import get_default_4_cohort_dataframes, extract_key_from_df


def generate_single_file_plot_dict(df, cohort_label="Batch", settings=None, dpi=140) -> dict:
    """
    Renders the complete set of publication-grade single-file figures in-memory
    for an individual dataset DataFrame.

    Returns:
        dict of {filename: png_bytes}
    """
    figs_dict = {}
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return figs_dict

    try:
        abbott_c = float(settings.get('abbott_c', 0.0)) if settings else 0.0
    except Exception:
        abbott_c = 0.0

    c = compute_single_batch_data(df, abbott_c=abbott_c)

    clean_safe_lbl = re.sub(r'[\/\\:\*\?"<>\|]', '_', str(cohort_label))
    days = c.get('elapsed_days', [])
    num_days = c.get('num_days', len(days))
    if num_days == 0:
        return figs_dict

    # 1. Probit Regression Plot
    try:
        fig_p, ax_p = plt.subplots(figsize=(7.5, 5.2), dpi=dpi)
        ax_p.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', zorder=1)
        log_t = np.array(c.get('log_time', []), dtype=float)
        prob_y = np.array(c.get('probit_y', []), dtype=float)
        a_val = float(c.get('probit_a', 0.0))
        b_val = float(c.get('probit_b', 0.0))
        r_val = float(c.get('probit_r', 0.0))
        r2_val = r_val ** 2
        n_pts = len(log_t)
        df_regr = max(1, n_pts - 2)

        if n_pts > 0:
            ax_p.scatter(log_t, prob_y, color='#0284c7', edgecolor='black', s=60, linewidth=1.1, zorder=4, label='Observed Probit')
            x_min = min(log_t) - 0.05 if len(log_t) else -0.25
            x_max = max(log_t) + 0.08 if len(log_t) else 1.25
            x_fit = np.linspace(min(-0.25, x_min), max(1.25, x_max), 100)
            y_fit = a_val + b_val * x_fit
            ax_p.plot(x_fit, y_fit, color='#0284c7', linewidth=2.2, zorder=3, label='Probit Fit')
            if n_pts > 2:
                y_pred = a_val + b_val * log_t
                resids = prob_y - y_pred
                s_err = np.sqrt(np.sum(resids**2) / df_regr)
                x_bar = np.mean(log_t)
                ss_x = np.sum((log_t - x_bar)**2)
                se_band = s_err * np.sqrt(1.0 / n_pts + (x_fit - x_bar)**2 / ss_x) if ss_x > 1e-9 else np.full_like(x_fit, 0.1)
                ax_p.fill_between(x_fit, y_fit - 1.96 * se_band, y_fit + 1.96 * se_band, color='#0284c7', alpha=0.15, zorder=2, label='95% CI Ribbon')

        ax_p.axhline(5.0, color='#64748b', linestyle='--', linewidth=1.4, zorder=2, label='LT50 Reference (P=5.0)')
        lt50_str = "Not Reached"
        if b_val > 1e-3:
            log_lt50 = (5.0 - a_val) / b_val
            lt50_days = 10.0**log_lt50
            if np.isfinite(lt50_days) and lt50_days > 0:
                lt50_str = f"{lt50_days:.1f} Days"
                ax_p.axvline(log_lt50, color='#dc2626', linestyle=':', linewidth=1.2, zorder=2)
                ax_p.plot([log_lt50], [5.0], marker='*', color='#dc2626', markersize=11, markeredgecolor='black', zorder=5)

        ax_p.set_title(f"Probit Regression Analysis: {cohort_label}", fontsize=12.5, fontweight='bold', pad=12)
        ax_p.set_xlabel("Log10(Observation Time / Days)", fontsize=11, fontweight='bold', labelpad=7)
        ax_p.set_ylabel("Probit Value (P)", fontsize=11, fontweight='bold', labelpad=7)
        ax_p.set_ylim(2.5, 9.2)
        ax_p.text(0.04, 0.95, f"Y = {a_val:.2f} + {b_val:.2f}x\nR² = {r2_val:.3f} | df = {df_regr}\nLT50: {lt50_str}",
                  transform=ax_p.transAxes, ha='left', va='top', fontsize=8.5, fontweight='bold',
                  bbox=dict(facecolor='white', edgecolor='#cbd5e1', alpha=0.90, boxstyle='round,pad=0.3'))
        ax_p.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=8.5)
        fig_p.tight_layout()
        buf_p = io.BytesIO()
        fig_p.savefig(buf_p, format='png', dpi=dpi)
        plt.close(fig_p)
        figs_dict['01_probit_regression.png'] = buf_p.getvalue()
    except Exception:
        pass

    # 2. Abbott Corrected Mortality Curve
    try:
        fig_m, ax_m = plt.subplots(figsize=(7.5, 5.2), dpi=dpi)
        ax_m.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', zorder=1)
        m_pts = np.array(c.get('abbott_corrected_pct', []), dtype=float)
        se_pts = np.array(c.get('mortality_se', np.zeros_like(m_pts)), dtype=float)
        if len(days) > 0 and len(m_pts) == len(days):
            ci_l = np.clip(m_pts - 1.96 * se_pts, 0, 100)
            ci_u = np.clip(m_pts + 1.96 * se_pts, 0, 100)
            ax_m.fill_between(days, ci_l, ci_u, color='#0284c7', alpha=0.15, zorder=2, label='95% Confidence Interval')
            ax_m.errorbar(days, m_pts, yerr=se_pts, fmt='o', color='#0284c7', ecolor='#0284c7',
                          elinewidth=1.4, capsize=3.5, capthick=1.2, markersize=7, markeredgecolor='black',
                          label='Abbott Corrected (Mean ± SE)', zorder=4)
            ax_m.plot(days, m_pts, color='#0284c7', linewidth=2.2, zorder=3)
        ax_m.axhline(50.0, color='#64748b', linestyle=':', linewidth=1.4, label='50% Threshold')
        ax_m.set_title(f"Abbott Corrected Mortality: {cohort_label}", fontsize=12.5, fontweight='bold', pad=12)
        ax_m.set_xlabel("Observation Time / Days", fontsize=11, fontweight='bold', labelpad=7)
        ax_m.set_ylabel("Corrected Mortality / % (Mean ± SE)", fontsize=11, fontweight='bold', labelpad=7)
        ax_m.set_ylim(-2, 105)
        ax_m.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=8.5)
        fig_m.tight_layout()
        buf_m = io.BytesIO()
        fig_m.savefig(buf_m, format='png', dpi=dpi)
        plt.close(fig_m)
        figs_dict['02_abbott_mortality_curve.png'] = buf_m.getvalue()
    except Exception:
        pass

    # 3. Kaplan-Meier Survival Analysis
    try:
        fig_km, ax_km = plt.subplots(figsize=(7.5, 5.2), dpi=dpi)
        ax_km.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', zorder=1)
        kmf = c.get('kmf')
        durations = np.array(c.get('durations', []))
        events = np.array(c.get('events', []))
        if kmf is not None and hasattr(kmf, 'survival_function_'):
            km_timeline = kmf.survival_function_.index
            km_vals = kmf.survival_function_['KM_estimate'].values
            ci_df = kmf.confidence_interval_survival_function_
            ax_km.step(km_timeline, km_vals, where='post', color='#0284c7', linewidth=2.4, label='Kaplan-Meier Estimate', zorder=3)
            ax_km.fill_between(km_timeline, ci_df.iloc[:, 0], ci_df.iloc[:, 1], step='post', color='#0284c7', alpha=0.15, label='95% Greenwood CI', zorder=2)
            cens_days = durations[events == 0]
            if len(cens_days) > 0:
                cens_surv = [kmf.survival_function_at_times(d).values[0] for d in cens_days]
                ax_km.plot(cens_days, cens_surv, '+', color='#0f172a', markersize=9, markeredgewidth=2, label=f'Censored (n={len(cens_days)})', zorder=4)
        ax_km.axhline(0.5, color='#64748b', linestyle=':', linewidth=1.4, label='50% Threshold')
        df_km = max(1, len(durations) - 1)
        ax_km.text(0.96, 0.95, f"N = {len(durations)} | df = {df_km}\nSE: Greenwood formula\n95% CI: log-log bounds",
                   transform=ax_km.transAxes, ha='right', va='top', fontsize=8.5, fontweight='bold',
                   bbox=dict(facecolor='white', edgecolor='#cbd5e1', alpha=0.90))
        ax_km.set_title(f"Kaplan-Meier Survival Analysis: {cohort_label}", fontsize=12.5, fontweight='bold', pad=12)
        ax_km.set_xlabel("Observation Time / Days", fontsize=11, fontweight='bold', labelpad=7)
        ax_km.set_ylabel("Survival Probability", fontsize=11, fontweight='bold', labelpad=7)
        ax_km.set_ylim(-0.02, 1.05)
        ax_km.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=8.5)
        fig_km.tight_layout()
        buf_km = io.BytesIO()
        fig_km.savefig(buf_km, format='png', dpi=dpi)
        plt.close(fig_km)
        figs_dict['03_kaplan_meier_survival.png'] = buf_km.getvalue()
    except Exception:
        pass

    # 4. Publication Composite 4-Panel Figure
    try:
        fig_pub = plt.figure(figsize=(13, 10), dpi=dpi)
        gs = fig_pub.add_gridspec(2, 2, hspace=0.32, wspace=0.26)

        # Panel A: Abbott Mortality
        ax_a = fig_pub.add_subplot(gs[0, 0])
        ax_a.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1')
        if len(days) > 0 and len(m_pts) == len(days):
            ax_a.errorbar(days, m_pts, yerr=se_pts, fmt='o-', color='#dc2626', ecolor='#dc2626', capsize=3.0, elinewidth=1.2, markersize=6)
        ax_a.axhline(50.0, color='#64748b', linestyle=':', linewidth=1.2)
        ax_a.set_title("A: Abbott Corrected Mortality", fontsize=11.5, fontweight='bold')
        ax_a.set_xlabel("Time / Days", fontsize=10, fontweight='bold')
        ax_a.set_ylabel("Mortality / % (Mean ± SE)", fontsize=10, fontweight='bold')
        ax_a.set_ylim(-2, 105)

        # Panel B: Probit Regression
        ax_b = fig_pub.add_subplot(gs[0, 1])
        ax_b.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1')
        if len(log_t) > 0:
            ax_b.scatter(log_t, prob_y, color='#0284c7', s=45, edgecolor='black', zorder=4)
            x_f = np.linspace(min(-0.25, min(log_t)-0.05), max(1.25, max(log_t)+0.05), 50)
            ax_b.plot(x_f, a_val + b_val * x_f, color='#0284c7', linewidth=2.0, zorder=3)
        ax_b.axhline(5.0, color='#64748b', linestyle='--', linewidth=1.2)
        ax_b.set_title(f"B: Probit Fit (Y = {a_val:.2f} + {b_val:.2f}x, R²={r2_val:.3f})", fontsize=11.5, fontweight='bold')
        ax_b.set_xlabel("Log10(Time / Days)", fontsize=10, fontweight='bold')
        ax_b.set_ylabel("Probit Value (P)", fontsize=10, fontweight='bold')
        ax_b.set_ylim(2.5, 9.2)

        # Panel C: Kaplan-Meier Survival
        ax_c = fig_pub.add_subplot(gs[1, 0])
        ax_c.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1')
        if kmf is not None and hasattr(kmf, 'survival_function_'):
            kmf.plot_survival_function(ax=ax_c, color='#059669', ci_show=True, ci_alpha=0.15, linewidth=2.2)
        ax_c.axhline(0.5, color='#64748b', linestyle=':', linewidth=1.2)
        ax_c.set_title("C: Kaplan-Meier Survival", fontsize=11.5, fontweight='bold')
        ax_c.set_xlabel("Time / Days", fontsize=10, fontweight='bold')
        ax_c.set_ylabel("Survival Probability", fontsize=10, fontweight='bold')
        ax_c.set_ylim(-0.02, 1.05)

        # Panel D: Environmental Temperature or Feeding
        ax_d = fig_pub.add_subplot(gs[1, 1])
        ax_d.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1')
        temps = c.get('temps', [])
        if any(temps):
            ax_d.plot(days, temps, 'o-', color='#d97706', linewidth=2.0, markersize=5.5)
            ax_d.set_title("D: Incubator Temperature Profile", fontsize=11.5, fontweight='bold')
            ax_d.set_xlabel("Time / Days", fontsize=10, fontweight='bold')
            ax_d.set_ylabel("Temperature / °C", fontsize=10, fontweight='bold')
        else:
            surv_pts = c.get('survival_pct', [])
            ax_d.plot(days, surv_pts, 's-', color='#7c3aed', linewidth=2.0, markersize=5.5)
            ax_d.set_title("D: Empirical Survival Rate", fontsize=11.5, fontweight='bold')
            ax_d.set_xlabel("Time / Days", fontsize=10, fontweight='bold')
            ax_d.set_ylabel("Survival / %", fontsize=10, fontweight='bold')

        fig_pub.suptitle(f"Publication 4-Panel Summary: {cohort_label}", fontsize=14, fontweight='bold', y=0.98)
        fig_pub.tight_layout(rect=[0, 0, 1, 0.96])
        buf_pub = io.BytesIO()
        fig_pub.savefig(buf_pub, format='png', dpi=dpi)
        plt.close(fig_pub)
        figs_dict['04_publication_4panel_figure.png'] = buf_pub.getvalue()
    except Exception:
        pass

    # 5. Lethal Times Profile Bar Chart
    try:
        from modules.parser import calculate_lt_percentiles
        lt_recs = calculate_lt_percentiles(
            intercept_a=a_val, slope_b=b_val, start_time=1.0,
            times_numeric=days, mortality_pct=m_pts,
            percentiles=[10, 20, 30, 50, 90]
        )
        if lt_recs:
            fig_lt, ax_lt = plt.subplots(figsize=(7.5, 4.8), dpi=dpi)
            ax_lt.grid(True, linestyle='--', alpha=0.35, axis='x', color='#cbd5e1')
            lbls = [r['label'] for r in lt_recs]
            vals = [r['effective_elapsed_int'] if np.isfinite(r.get('effective_elapsed_int', np.nan)) else 0.0 for r in lt_recs]
            y_pos = np.arange(len(lbls))
            for i, r in enumerate(lt_recs):
                v = vals[i]
                if v > 0:
                    ax_lt.plot([0, v], [i, i], color='#0284c7', linewidth=2.5, zorder=2)
                    ax_lt.plot(v, i, marker='o', markersize=9, color='#0284c7', markeredgecolor='black', zorder=3)
                    ax_lt.text(v + 0.25, i, f"{int(round(v))} Days ({r['status']})", va='center', ha='left', fontsize=9.5, fontweight='bold')
                else:
                    ax_lt.plot([0, 0.4], [i, i], color='#94a3b8', linestyle=':', linewidth=1.5, zorder=2)
                    ax_lt.text(0.55, i, "Not Reached in Trial", va='center', ha='left', fontsize=9.0, fontstyle='italic', color='#64748b')
            ax_lt.set_yticks(y_pos)
            ax_lt.set_yticklabels(lbls, fontsize=10.5, fontweight='bold')
            ax_lt.set_xlabel("Elapsed Time / Days", fontsize=11, fontweight='bold')
            ax_lt.set_title(f"Lethal Time (LTₓ) Profile: {cohort_label}", fontsize=12.5, fontweight='bold', pad=12)
            max_v = max([v for v in vals if np.isfinite(v)], default=5.0)
            ax_lt.set_xlim(-0.2, max_v + 3.5)
            fig_lt.tight_layout()
            buf_lt = io.BytesIO()
            fig_lt.savefig(buf_lt, format='png', dpi=dpi)
            plt.close(fig_lt)
            figs_dict['05_lethal_times_profile.png'] = buf_lt.getvalue()
    except Exception:
        pass

    # 6. Daily Sucrose Consumption (if consumption columns exist)
    try:
        if c.get('has_consumption') and c.get('daily_mean_cons'):
            fig_c, ax_c = plt.subplots(figsize=(7.5, 4.8), dpi=dpi)
            ax_c.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1')
            mean_cons = np.array(c['daily_mean_cons'], dtype=float)
            sem_cons = np.array(c.get('daily_sem_cons', np.zeros_like(mean_cons)), dtype=float)
            x_days = np.arange(1, len(mean_cons) + 1)
            ax_c.fill_between(x_days, np.clip(mean_cons - sem_cons, 0, None), mean_cons + sem_cons, color='#d97706', alpha=0.15)
            ax_c.errorbar(x_days, mean_cons, yerr=sem_cons, fmt='o-', color='#d97706', ecolor='#d97706', capsize=3.0, elinewidth=1.2, markersize=6.5, markeredgecolor='black')
            ax_c.set_title(f"Daily Sucrose Intake: {cohort_label}", fontsize=12.5, fontweight='bold', pad=12)
            ax_c.set_xlabel("Observation Day", fontsize=11, fontweight='bold')
            ax_c.set_ylabel("Sucrose Consumption (µL / bee / day)", fontsize=11, fontweight='bold')
            fig_c.tight_layout()
            buf_c = io.BytesIO()
            fig_c.savefig(buf_c, format='png', dpi=dpi)
            plt.close(fig_c)
            figs_dict['06_daily_sucrose_consumption.png'] = buf_c.getvalue()
    except Exception:
        pass

    # 7. Raw CSV data
    figs_dict[f"data_{clean_safe_lbl}.csv"] = df.to_csv(index=False).encode('utf-8')

    return figs_dict


def get_all_active_batches(uploaded_file_dict=None):
    """
    Returns an ordered dictionary of {filename: (cohort_display_label, dataframe)}
    for all currently active datasets (from session upload, Tab 1, or defaults).
    """
    batches = {}
    uploaded_dict = uploaded_file_dict if uploaded_file_dict is not None else st.session_state.get('uploaded_file_dict', {})
    if uploaded_dict:
        for fname, file_data in uploaded_dict.items():
            if isinstance(file_data, tuple):
                c_name, c_df = file_data
            elif isinstance(file_data, dict):
                c_name = file_data.get('name', fname)
                c_df = file_data.get('df')
            else:
                c_name = str(fname)
                c_df = file_data

            key_val = extract_key_from_df(c_df)
            lbl = key_val if key_val else (c_name if c_name else fname)
            batches[fname] = (lbl, c_df)
    else:
        # Fallback to active observation sheet in Tab 1
        active_df = st.session_state.get('df')
        if active_df is not None and isinstance(active_df, pd.DataFrame) and not active_df.empty:
            active_name = st.session_state.get('active_sheet_key', 'Active_Observation_Sheet')
            batches['single_active_batch.csv'] = (active_name, active_df)
        else:
            # Fallback to the 4 standard agreed default cohorts
            for c_name, c_df in get_default_4_cohort_dataframes().items():
                safe_fn = c_name.replace(' ', '_').replace('%', 'pct').replace('µL', 'uL') + '.csv'
                batches[safe_fn] = (c_name, c_df)

    return batches


def get_single_plots_zip(settings=None, uploaded_file_dict=None) -> bytes:
    """
    Builds a ZIP package containing the complete suite of single-file plots
    for EVERY uploaded/active dataset (each structured in its own folder).
    """
    if settings is None:
        settings = st.session_state.get('studio_settings', {})
    batches = get_all_active_batches(uploaded_file_dict=uploaded_file_dict)
    if not batches:
        return b""

    buf = io.BytesIO()
    total_figs = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, (fname, (lbl, df)) in enumerate(batches.items()):
            clean_name = re.sub(r'[/\\:*?"<>|]', '_', str(fname).replace('.csv', ''))
            folder_tag = f"dataset_{idx+1}_{clean_name}"
            single_figs = generate_single_file_plot_dict(df, cohort_label=lbl, settings=settings)
            for fig_name, fig_bytes in single_figs.items():
                zf.writestr(f"single_file_plots/{folder_tag}/{fig_name}", fig_bytes)
                if fig_name.endswith('.png'):
                    total_figs += 1

        manifest = (
            "Bee Analytics Studio — All Single-File Plots Package\n"
            "==============================================================\n\n"
            f"• Number of Datasets: {len(batches)}\n"
            f"• Total Single-File Figures: {total_figs} plots\n"
            "• Resolution: Publication-grade 140–300 DPI\n"
            "• Structure: Each dataset contains its own Probit regression, Abbott mortality,\n"
            "  Kaplan-Meier survival, 4-panel publication composite, and lethal time spectrum.\n"
        )
        zf.writestr("MANIFEST.txt", manifest.encode('utf-8'))

    return buf.getvalue()


def get_multi_plots_zip() -> bytes:
    """Retrieves or builds the ZIP package containing all multi-CSV comparison plots."""
    multi_images = st.session_state.get('multi_plot_images', {})
    if not multi_images:
        return b""

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, data in multi_images.items():
            zf.writestr(fname, data)
        # Include summary dataframe if available
        summary_df = st.session_state.get('multi_summary_df')
        if summary_df is not None and isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
            zf.writestr("multi_csv_probit_summary.csv", summary_df.to_csv(index=False).encode('utf-8'))
    return buf.getvalue()


def get_complete_master_zip(settings=None, uploaded_file_dict=None, multi_images=None) -> bytes:
    """
    Builds the comprehensive master ZIP archive bundling:
    1. multi_cohort_comparison/ : All comparative Multi-CSV plots & summary CSV
    2. single_file_plots/ : A dedicated folder for EVERY uploaded file with all its individual plots
    3. Complete manifest summary.
    """
    if settings is None:
        settings = st.session_state.get('studio_settings', {})
    if multi_images is None:
        multi_images = st.session_state.get('multi_plot_images', {})
    batches = get_all_active_batches(uploaded_file_dict=uploaded_file_dict)

    if not multi_images and not batches:
        return b""

    buf = io.BytesIO()
    total_single_figs = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Multi-Cohort Comparison Folder
        for fname, data in multi_images.items():
            zf.writestr(f"multi_cohort_comparison/{fname}", data)
        summary_df = st.session_state.get('multi_summary_df')
        if summary_df is not None and isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
            zf.writestr("multi_cohort_comparison/multi_csv_probit_summary.csv", summary_df.to_csv(index=False).encode('utf-8'))

        # 2. Single File Plots Folders (for EVERY active uploaded dataset)
        for idx, (fname, (lbl, df)) in enumerate(batches.items()):
            clean_name = re.sub(r'[/\\:*?"<>|]', '_', str(fname).replace('.csv', ''))
            folder_tag = f"dataset_{idx+1}_{clean_name}"
            single_figs = generate_single_file_plot_dict(df, cohort_label=lbl, settings=settings)
            for fig_name, fig_bytes in single_figs.items():
                zf.writestr(f"single_file_plots/{folder_tag}/{fig_name}", fig_bytes)
                if fig_name.endswith('.png'):
                    total_single_figs += 1

        # 3. Readme Manifest summary text inside the ZIP
        manifest_text = (
            "Bee Analytics Studio — Complete Master Plot & Data Archive\n"
            "==============================================================\n\n"
            f"• Number of Uploaded/Active Datasets: {len(batches)}\n"
            f"• Single-File Analysis Figures: {total_single_figs} plots across {len(batches)} dataset folders\n"
            f"• Multi-Cohort Comparison Figures: {len(multi_images)} plots\n"
            "• Resolution: Publication-grade 140–300 DPI\n"
            "• Bioassay Protocol: Post-2h Starvation Rule Compliant (Day 0 Baseline Excluded)\n\n"
            "DIRECTORY STRUCTURE:\n"
            "├── multi_cohort_comparison/     (Cross-cohort comparative Probit, Mortality, Survival, LT50 bars, and Summary CSV)\n"
            "└── single_file_plots/           (Dedicated subfolder for every single file containing individual Probit,\n"
            "                                 Abbott mortality, Kaplan-Meier, 4-panel figure, LT profile, and raw CSV)\n"
        )
        zf.writestr("MANIFEST_SUMMARY.txt", manifest_text.encode('utf-8'))

    return buf.getvalue()


def render_sidebar_global_download(settings=None):
    """Renders the persistent global download center inside the Streamlit sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h3 class='sidebar-h3'>📦 Global Download Center</h3>", unsafe_allow_html=True)
    st.sidebar.caption("Download all figures and summaries across all uploaded files in publication resolution.")

    if settings is None:
        settings = st.session_state.get('studio_settings', {})

    batches = get_all_active_batches()
    n_files = len(batches)
    multi_images = st.session_state.get('multi_plot_images', {})
    n_multi = len(multi_images)

    single_bytes = get_single_plots_zip(settings=settings)
    multi_bytes = get_multi_plots_zip()
    complete_bytes = get_complete_master_zip(settings=settings)

    if complete_bytes:
        st.sidebar.download_button(
            label=f"🌟 Complete Master Bundle ({n_files} Datasets: Multi + All Single Plots)",
            data=complete_bytes,
            file_name="complete_bee_analytics_master_bundle.zip",
            mime="application/zip",
            use_container_width=True,
            key="sb_dl_complete_master"
        )

    if single_bytes:
        st.sidebar.download_button(
            label=f"📦 All Single-File Plots ({n_files} Datasets)",
            data=single_bytes,
            file_name="all_single_file_plots.zip",
            mime="application/zip",
            use_container_width=True,
            key="sb_dl_single_all"
        )

    if multi_bytes:
        st.sidebar.download_button(
            label=f"📊 Multi-Cohort Comparison Plots ({n_multi} Figs)",
            data=multi_bytes,
            file_name="all_multi_cohort_plots.zip",
            mime="application/zip",
            use_container_width=True,
            key="sb_dl_multi_all"
        )


def render_header_global_download(settings=None):
    """Renders an interactive quick-download bar directly in the top header."""
    if settings is None:
        settings = st.session_state.get('studio_settings', {})

    batches = get_all_active_batches()
    n_files = len(batches)
    multi_images = st.session_state.get('multi_plot_images', {})
    n_multi = len(multi_images)

    single_bytes = get_single_plots_zip(settings=settings)
    multi_bytes = get_multi_plots_zip()
    complete_bytes = get_complete_master_zip(settings=settings)

    if not single_bytes and not multi_bytes and not complete_bytes:
        return

    st.markdown("""
<div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 12px; padding: 10px 18px; margin-bottom: 14px;">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
        <span style="font-weight: 700; color: #b45309; font-size: 13.5px;">⚡ Global One-Click Figure Exports (Publication Ready ZIP Archives):</span>
    </div>
</div>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        if single_bytes:
            st.download_button(
                label=f"📦 All Single Plots ({n_files} Datasets)",
                data=single_bytes,
                file_name="all_single_file_plots.zip",
                mime="application/zip",
                use_container_width=True,
                key="hdr_dl_single_all"
            )
        else:
            st.button("📦 Single Plots (Upload Files)", disabled=True, use_container_width=True, key="hdr_dl_single_dis")

    with col2:
        if multi_bytes:
            st.download_button(
                label=f"📊 Multi Plots ({n_multi} Figs)",
                data=multi_bytes,
                file_name="all_multi_cohort_plots.zip",
                mime="application/zip",
                use_container_width=True,
                key="hdr_dl_multi_all"
            )
        else:
            st.button("📊 Multi Plots (Open Tab 2)", disabled=True, use_container_width=True, key="hdr_dl_multi_dis")

    with col3:
        if complete_bytes:
            st.download_button(
                label=f"🌟 Complete Master Bundle ({n_files} Files: Multi + Singles)",
                data=complete_bytes,
                file_name="complete_bee_analytics_master_bundle.zip",
                mime="application/zip",
                use_container_width=True,
                key="hdr_dl_master_all"
            )
        else:
            st.button("🌟 Complete Bundle (Awaiting Plots)", disabled=True, use_container_width=True, key="hdr_dl_master_dis")


