"""
Multiple Testing Correction & False Discovery Rate (FDR) Engine for Bioassay Analysis.
Implements rigorous statistical multiplicity adjustments:
  1. Bonferroni Single-Step (FWER Control)
  2. Holm-Bonferroni Step-Down (FWER Control)
  3. Benjamini-Hochberg (BH Step-Up) (FDR Control: E[V/R] <= q)
  4. Benjamini-Yekutieli (BY Step-Up) (FDR Control under arbitrary dependence)
  5. Šidák Single-Step (FWER Control under independence)

Provides complete theoretical confusion matrix accounting:
  - V: False Positives (Type I error)
  - T: False Negatives (Type II error)
  - S: True Positives
  - U: True Negatives
  - R: Total Rejections (R = V + S)
  - FDR = E[V / R] (where V/R = 0 if R = 0)
  - FWER = P(V >= 1)
"""
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    import statsmodels.stats.multitest as smm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def compute_multiple_corrections(p_values_dict: dict, alpha: float = 0.05, fdr_q: float = 0.05) -> pd.DataFrame:
    """
    Computes comprehensive multiple testing corrections across a dictionary of {comparison_label: raw_p_value}.

    Returns:
        pd.DataFrame with raw and adjusted p-values (Bonferroni, Holm, BH, BY, Šidák)
        and significance flags at target alpha and FDR q.
    """
    if not p_values_dict:
        return pd.DataFrame()

    labels = list(p_values_dict.keys())
    raw_p = np.array([float(p_values_dict[k]) for k in labels], dtype=float)
    # Clamp p-values to [1e-15, 1.0] for stability
    raw_p = np.clip(raw_p, 1e-15, 1.0)
    m = len(raw_p)

    if HAS_STATSMODELS:
        _, p_bonf, _, _ = smm.multipletests(raw_p, alpha=alpha, method='bonferroni')
        _, p_holm, _, _ = smm.multipletests(raw_p, alpha=alpha, method='holm')
        _, q_bh, _, _ = smm.multipletests(raw_p, alpha=fdr_q, method='fdr_bh')
        _, q_by, _, _ = smm.multipletests(raw_p, alpha=fdr_q, method='fdr_by')
        _, p_sidak, _, _ = smm.multipletests(raw_p, alpha=alpha, method='sidak')
    else:
        # Vectorized pure-Python fallback
        # 1. Bonferroni: p_adj = min(1, m * p)
        p_bonf = np.minimum(1.0, raw_p * m)

        # 2. Holm-Bonferroni step-down
        order = np.argsort(raw_p)
        sorted_p = raw_p[order]
        holm_factors = m - np.arange(m)
        sorted_holm = np.minimum(1.0, np.maximum.accumulate(sorted_p * holm_factors))
        p_holm = np.zeros(m)
        p_holm[order] = sorted_holm

        # 3. Benjamini-Hochberg (BH) FDR: q = min_{j>=i} (m/j * p_(j))
        bh_factors = m / (np.arange(1, m + 1))
        sorted_bh = sorted_p * bh_factors
        # Running minimum backwards
        sorted_bh_q = np.minimum.accumulate(sorted_bh[::-1])[::-1]
        sorted_bh_q = np.minimum(1.0, sorted_bh_q)
        q_bh = np.zeros(m)
        q_bh[order] = sorted_bh_q

        # 4. Benjamini-Yekutieli (BY) FDR: factor = m * c(m) / j
        cm = np.sum(1.0 / np.arange(1, m + 1))
        by_factors = (m * cm) / (np.arange(1, m + 1))
        sorted_by = sorted_p * by_factors
        sorted_by_q = np.minimum.accumulate(sorted_by[::-1])[::-1]
        sorted_by_q = np.minimum(1.0, sorted_by_q)
        q_by = np.zeros(m)
        q_by[order] = sorted_by_q

        # 5. Šidák: 1 - (1 - p)^m
        p_sidak = 1.0 - (1.0 - raw_p) ** m

    rows = []
    for i in range(m):
        rows.append({
            "Comparison": labels[i],
            "Raw p-value": raw_p[i],
            "Bonferroni p_adj": p_bonf[i],
            "Holm p_adj": p_holm[i],
            "BH FDR q-value": q_bh[i],
            "BY FDR q-value": q_by[i],
            "Šidák p_adj": p_sidak[i],
            "Sig (Raw < 0.05)": bool(raw_p[i] < alpha),
            "Sig (Bonferroni)": bool(p_bonf[i] < alpha),
            "Sig (Holm)": bool(p_holm[i] < alpha),
            "Sig (BH FDR)": bool(q_bh[i] < fdr_q),
            "Sig (BY FDR)": bool(q_by[i] < fdr_q),
            "Sig (Šidák)": bool(p_sidak[i] < alpha),
        })

    df_out = pd.DataFrame(rows)
    return df_out


def generate_multi_testing_plot(df_corr: pd.DataFrame, alpha: float = 0.05, fdr_q: float = 0.05, dpi: int = 140):
    """
    Renders a publication-grade Multi-Correction P-value & FDR Comparison Plot.
    Plots raw vs Bonferroni, Holm, Benjamini-Hochberg, and Benjamini-Yekutieli adjusted values.
    """
    if df_corr.empty:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
        ax.text(0.5, 0.5, "No Pairwise Comparisons Available", ha='center', va='center')
        return fig

    m = len(df_corr)
    fig_w = max(9.5, m * 1.1 + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 5.8), dpi=dpi)
    ax.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', axis='y', zorder=1)

    x_indices = np.arange(m)
    labels = df_corr['Comparison'].values

    # Convert to -log10(p) for visual clarity, clamped at 10 for extreme values
    def safe_neg_log10(p_arr):
        return -np.log10(np.clip(p_arr, 1e-10, 1.0))

    raw_nlp = safe_neg_log10(df_corr['Raw p-value'].values)
    bonf_nlp = safe_neg_log10(df_corr['Bonferroni p_adj'].values)
    holm_nlp = safe_neg_log10(df_corr['Holm p_adj'].values)
    bh_nlp = safe_neg_log10(df_corr['BH FDR q-value'].values)
    by_nlp = safe_neg_log10(df_corr['BY FDR q-value'].values)

    # Offset markers slightly along x for clarity
    offsets = [-0.22, -0.11, 0.0, +0.11, +0.22]

    ax.scatter(x_indices + offsets[0], raw_nlp, color='#64748b', marker='o', s=65, edgecolor='black', linewidth=1.0, zorder=4, label='Raw p-value')
    ax.scatter(x_indices + offsets[1], bonf_nlp, color='#ef4444', marker='s', s=70, edgecolor='black', linewidth=1.0, zorder=4, label='Bonferroni (FWER)')
    ax.scatter(x_indices + offsets[2], holm_nlp, color='#f59e0b', marker='^', s=75, edgecolor='black', linewidth=1.0, zorder=4, label='Holm-Bonferroni (FWER)')
    ax.scatter(x_indices + offsets[3], bh_nlp, color='#10b981', marker='D', s=65, edgecolor='black', linewidth=1.0, zorder=4, label='Benjamini-Hochberg (BH FDR)')
    ax.scatter(x_indices + offsets[4], by_nlp, color='#8b5cf6', marker='v', s=70, edgecolor='black', linewidth=1.0, zorder=4, label='Benjamini-Yekutieli (BY FDR)')

    # Connect markers for each comparison with vertical stem lines
    for i in range(m):
        pts = [raw_nlp[i], bonf_nlp[i], holm_nlp[i], bh_nlp[i], by_nlp[i]]
        ax.plot([x_indices[i], x_indices[i]], [min(pts), max(pts)], color='#cbd5e1', linestyle=':', linewidth=1.2, zorder=2)

    # Threshold horizontal line at alpha = 0.05 (-log10(0.05) ≈ 1.301)
    thresh_nlp = -np.log10(alpha)
    ax.axhline(thresh_nlp, color='#dc2626', linestyle='--', linewidth=1.8, zorder=3, label=f'Significance Cutoff (α = {alpha}, -log₁₀p = {thresh_nlp:.2f})')

    ax.set_xticks(x_indices)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9.5, fontweight='bold')
    ax.set_ylabel('-Log₁₀(p-value / FDR q-value)', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title(f'Multiple Hypothesis Testing Corrections Across Pairwise Comparisons (m = {m})', fontsize=12.5, fontweight='bold', pad=12)

    # Highlight region above threshold
    y_max = max(max(raw_nlp), max(bonf_nlp), max(bh_nlp), thresh_nlp) * 1.15 + 0.3
    ax.set_ylim(-0.1, y_max)
    ax.axhspan(thresh_nlp, y_max, color='#10b981', alpha=0.04, zorder=1)

    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.92, edgecolor='#cbd5e1', fontsize=9.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig


def generate_pvalue_histogram(p_values_dict: dict, alpha: float = 0.05, dpi: int = 130) -> plt.Figure:
    """
    Renders a publication-grade frequency histogram of empirical p-values across all pairwise tests.
    Displays:
    - Binned distribution of p-values in [0, 1]
    - Expected null frequency line under H0: Uniform U(0, 1)
    - Highlighted significant bin [0, alpha)
    - Annotated summary metrics (m, R_sig, proportion sig, min p)
    """
    if not p_values_dict:
        fig, ax = plt.subplots(figsize=(6, 3), dpi=dpi)
        ax.text(0.5, 0.5, "No p-values available", ha='center', va='center')
        return fig

    p_arr = np.array([float(v) for v in p_values_dict.values()], dtype=float)
    p_arr = np.clip(p_arr, 0.0, 1.0)
    m = len(p_arr)

    n_bins = 10 if m <= 15 else 20
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_width = 1.0 / n_bins

    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=dpi)
    ax.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', axis='y', zorder=1)

    counts, edges = np.histogram(p_arr, bins=bin_edges)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    bar_colors = []
    for edge_left in edges[:-1]:
        if edge_left < alpha:
            bar_colors.append('#10b981' if np.any(p_arr < alpha) else '#ef4444')
        else:
            bar_colors.append('#64748b')

    bars = ax.bar(
        bin_centers, counts, width=bin_width * 0.90,
        color=bar_colors, edgecolor='black', linewidth=1.1, zorder=3,
        label=f'Pairwise Tests Frequency (m = {m})'
    )

    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.12, f"{int(h)}",
                    ha='center', va='bottom', fontsize=9.0, fontweight='bold', color='#1e293b')

    exp_h0 = m * bin_width
    ax.axhline(exp_h0, color='#0284c7', linestyle='--', linewidth=2.0, zorder=4,
               label=f'Expected under H₀ (Uniform U[0,1] = {exp_h0:.2f})')

    ax.axvline(alpha, color='#dc2626', linestyle=':', linewidth=2.0, zorder=4,
               label=f'Significance Cutoff (α = {alpha})')

    n_sig = int(np.sum(p_arr < alpha))
    pct_sig = (n_sig / m) * 100.0 if m > 0 else 0.0
    min_p = np.min(p_arr) if m > 0 else 1.0
    min_p_str = "< 0.0001" if min_p < 0.0001 else f"{min_p:.4f}"

    badge_lines = [
        f"Hypotheses Evaluated: m = {m}",
        f"Significant Tests (p < {alpha}): R = {n_sig} ({pct_sig:.1f}%)",
        f"Minimum p-value: {min_p_str}",
        f"Null Evaluation: {'Anti-Conservative Spike (True Signal)' if n_sig > exp_h0 else 'Uniform / Consistent with H₀'}"
    ]
    ax.text(0.97, 0.94, "\n".join(badge_lines), transform=ax.transAxes,
            ha='right', va='top', fontsize=9.0, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#cbd5e1', alpha=0.92))

    ax.set_xlabel('Empirical Pairwise p-value', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('Frequency (Number of Hypotheses)', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title(f'Pairwise Hypothesis Test p-value Distribution (α = {alpha})', fontsize=12.5, fontweight='bold', pad=12)
    ax.set_xlim(-0.02, 1.02)
    y_top = max(max(counts) if len(counts) > 0 else 1, exp_h0) * 1.35 + 1.0
    ax.set_ylim(0, y_top)

    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.92, edgecolor='#cbd5e1', fontsize=9.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig


def render_multiple_testing_math_card():
    """Renders the comprehensive theoretical confusion matrix and mathematical guide for Multiple Testing Corrections and FDR."""
    st.markdown("""
<div style="background: linear-gradient(135deg, rgba(248,250,252,0.95), rgba(241,245,249,0.95)); border: 1px solid #cbd5e1; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
    <h4 style="margin-top: 0; color: #0f172a; display: flex; align-items: center; gap: 8px;">
        📐 Multiple Hypothesis Testing & False Discovery Rate (FDR) Framework
    </h4>
    <p style="font-size: 0.9rem; color: #475569; margin-bottom: 12px;">
        When conducting multiple pairwise statistical tests simultaneously across bioassay cohorts, the cumulative probability of at least one false positive (Type I error) inflates rapidly without multiplicity correction.
    </p>

    <!-- Theoretical Confusion Matrix Table -->
    <div style="background: white; border-radius: 8px; padding: 14px; border-left: 4px solid #0284c7; margin-bottom: 14px; overflow-x: auto;">
        <b style="color: #0369a1; font-size: 0.95rem;">Theoretical Hypothesis Testing Outcomes ($2 \\times 2$ Confusion Matrix):</b>
        <table style="width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.88rem; text-align: center;">
            <thead>
                <tr style="background: #f1f5f9; border-bottom: 2px solid #cbd5e1;">
                    <th style="padding: 8px; text-align: left;">Truth \\ Decision</th>
                    <th style="padding: 8px; color: #dc2626;">Declared Significant (Reject $H_0$)</th>
                    <th style="padding: 8px; color: #059669;">Declared Non-Significant (Retain $H_0$)</th>
                    <th style="padding: 8px; font-weight: bold;">Total Hypotheses</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 8px; text-align: left; font-weight: bold;">$H_0$ True (No effect)</td>
                    <td style="padding: 8px; background: rgba(239,68,68,0.08); font-weight: bold; color: #b91c1c;">$V$ (False Positives / Type I)</td>
                    <td style="padding: 8px; background: rgba(16,185,129,0.08); font-weight: bold; color: #047857;">$U$ (True Negatives)</td>
                    <td style="padding: 8px; font-weight: bold;">$m_0$</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 8px; text-align: left; font-weight: bold;">$H_1$ True (Real effect)</td>
                    <td style="padding: 8px; background: rgba(16,185,129,0.08); font-weight: bold; color: #047857;">$S$ (True Positives)</td>
                    <td style="padding: 8px; background: rgba(245,158,11,0.08); font-weight: bold; color: #b45309;">$T$ (False Negatives / Type II)</td>
                    <td style="padding: 8px; font-weight: bold;">$m - m_0$</td>
                </tr>
                <tr style="background: #f8fafc; font-weight: bold;">
                    <td style="padding: 8px; text-align: left;">Total Rejections / Non-Rejections</td>
                    <td style="padding: 8px; color: #0369a1;">$R = V + S$ (Total Discoveries)</td>
                    <td style="padding: 8px; color: #64748b;">$m - R = U + T$</td>
                    <td style="padding: 8px; color: #0f172a;">$m$ (Total Tests)</td>
                </tr>
            </tbody>
        </table>
        <div style="font-size: 0.82rem; color: #64748b; margin-top: 8px; line-height: 1.5;">
            <b>Key Variables:</b> $m$ = total comparisons tested; $R$ = total rejected null hypotheses; $V$ = unobserved false positive discoveries; $S$ = true positive discoveries.
        </div>
    </div>

    <!-- Mathematical Definitions Card -->
    <div style="background: white; border-radius: 8px; padding: 12px; border-left: 4px solid #10b981; margin-bottom: 14px;">
        <b style="color: #047857;">Core Multiplicity Error Rate Metrics:</b>
        <div style="margin: 8px 0; font-size: 0.92rem; line-height: 1.8;">
            <b>1. False Discovery Rate (FDR):</b> Expected proportion of false discoveries among all rejected hypotheses:
            $$\\text{FDR} = E\\left[\\frac{V}{R} \\;\\middle|\\; R > 0\\right] \\cdot P(R > 0) = E\\left[\\frac{V}{\\max(R, 1)}\\right]$$
            <b>2. Family-Wise Error Rate (FWER):</b> Probability of making at least one Type I error across the family:
            $$\\text{FWER} = P(V \\ge 1) = 1 - (1 - \\alpha)^m \\approx m \\alpha$$
        </div>
    </div>

    <!-- Correction Methods Comparison -->
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px;">
        <div style="background: white; border-radius: 8px; padding: 10px; border-top: 3px solid #ef4444;">
            <b style="color: #b91c1c; font-size: 0.88rem;">Bonferroni (FWER)</b>
            <div style="font-size: 0.82rem; color: #334155; margin-top: 4px;">
                Single-step FWER control:<br>
                $$p_{\\text{adj}} = \\min(1, m \\cdot p)$$
                Significance if $p \\le \\alpha / m$. Highly conservative.
            </div>
        </div>
        <div style="background: white; border-radius: 8px; padding: 10px; border-top: 3px solid #f59e0b;">
            <b style="color: #b45309; font-size: 0.88rem;">Holm-Bonferroni (FWER)</b>
            <div style="font-size: 0.82rem; color: #334155; margin-top: 4px;">
                Step-down FWER control:<br>
                Sort $p_{(1)} \\le \\dots \\le p_{(m)}$.<br>
                Reject while $p_{(i)} \\le \\frac{\\alpha}{m - i + 1}$. Uniformly more powerful than Bonferroni.
            </div>
        </div>
        <div style="background: white; border-radius: 8px; padding: 10px; border-top: 3px solid #10b981;">
            <b style="color: #047857; font-size: 0.88rem;">Benjamini-Hochberg (BH FDR)</b>
            <div style="font-size: 0.82rem; color: #334155; margin-top: 4px;">
                Step-up FDR control ($E[V/R] \\le q$):<br>
                Find largest $k$ where $p_{(k)} \\le \\frac{k}{m} q$.<br>
                Rejects $H_{(1)}, \\dots, H_{(k)}$. Optimal balance of power & precision.
            </div>
        </div>
        <div style="background: white; border-radius: 8px; padding: 10px; border-top: 3px solid #8b5cf6;">
            <b style="color: #6d28d9; font-size: 0.88rem;">Benjamini-Yekutieli (BY FDR)</b>
            <div style="font-size: 0.82rem; color: #334155; margin-top: 4px;">
                Arbitrary dependence FDR control:<br>
                $$p_{(k)} \\le \\frac{k}{m \\cdot c(m)} q, \\quad c(m) = \\sum_{j=1}^m \\frac{1}{j}$$
                Guaranteed FDR under arbitrary correlation structures.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


def render_multiple_testing_studio(p_values_dict: dict, key_suffix: str = "", image_collector=None):
    """
    Renders the interactive Multiple Testing Correction & FDR Studio component:
    - Confusion Matrix & FDR formula display
    - KPI cards: m hypotheses, R raw, R Bonferroni, R BH FDR
    - Multi-method comparison table with CSV download
    - Multi-correction Comparison Plot with PNG download
    """
    if not p_values_dict:
        st.info("No pairwise hypothesis tests available for multiple testing correction.")
        return

    m = len(p_values_dict)
    st.markdown("### 🔬 Multiple Hypothesis Testing & FDR Correction Suite")
    st.caption("Post-hoc multiplicity adjustments controlling Family-Wise Error Rate (FWER) and False Discovery Rate (FDR = E[V/R]) across all pairwise comparisons.")

    # Educational Math Card
    render_multiple_testing_math_card()

    # Interactive Threshold Controls
    c_ctl1, c_ctl2, c_ctl3 = st.columns([1, 1, 1.2])
    with c_ctl1:
        alpha_val = st.slider("FWER Significance Level (α)", min_value=0.01, max_value=0.10, value=0.05, step=0.01, key=f"mt_alpha_{key_suffix}")
    with c_ctl2:
        fdr_val = st.slider("FDR Target Discovery Rate (q)", min_value=0.01, max_value=0.20, value=0.05, step=0.01, key=f"mt_q_{key_suffix}")
    with c_ctl3:
        method_view = st.selectbox(
            "Primary Decision Method to Highlight:",
            ["Benjamini-Hochberg (BH FDR) [Recommended]", "Bonferroni (FWER)", "Holm-Bonferroni (FWER)", "Benjamini-Yekutieli (BY FDR)", "Šidák (FWER)"],
            index=0,
            key=f"mt_method_{key_suffix}"
        )

    # Compute corrections
    df_corr = compute_multiple_corrections(p_values_dict, alpha=alpha_val, fdr_q=fdr_val)
    if df_corr.empty:
        return

    r_raw = int(np.sum(df_corr['Sig (Raw < 0.05)']))
    r_bonf = int(np.sum(df_corr['Sig (Bonferroni)']))
    r_holm = int(np.sum(df_corr['Sig (Holm)']))
    r_bh = int(np.sum(df_corr['Sig (BH FDR)']))
    r_by = int(np.sum(df_corr['Sig (BY FDR)']))

    # KPI Cards
    col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
    with col_k1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{m}</div><div class="kpi-label">Hypotheses (m)</div></div>', unsafe_allow_html=True)
    with col_k2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color: #64748b;">{r_raw}</div><div class="kpi-label">Raw Rejections (R)</div></div>', unsafe_allow_html=True)
    with col_k3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color: #ef4444;">{r_bonf}</div><div class="kpi-label">Bonferroni Sig (FWER)</div></div>', unsafe_allow_html=True)
    with col_k4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color: #10b981;">{r_bh}</div><div class="kpi-label">BH Discoveries (FDR)</div></div>', unsafe_allow_html=True)
    with col_k5:
        # Theoretical upper bound on expected false discoveries: E[V] <= q * R
        ev_upper = f"{fdr_val * r_bh:.2f}" if r_bh > 0 else "0.00"
        st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color: #0284c7;">≤ {ev_upper}</div><div class="kpi-label">Expected False Pos (E[V])</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Plot & Table Tabs
    tab_plt, tab_hist, tab_tbl = st.tabs([
        "📊 Multi-Correction P-value & FDR Comparison Plot",
        "📉 Pairwise P-Value Distribution Histogram",
        "📋 Full Multi-Correction Decision Table (CSV Export)"
    ])

    with tab_plt:
        fig_mt = generate_multi_testing_plot(df_corr, alpha=alpha_val, fdr_q=fdr_val, dpi=130)
        st.pyplot(fig_mt)

        buf_mt = io.BytesIO()
        fig_mt.savefig(buf_mt, format='png', dpi=300, bbox_inches='tight')
        plt_bytes = buf_mt.getvalue()
        plt.close(fig_mt)

        if image_collector:
            image_collector('multiple_testing_correction_pvalues.png', plt_bytes)
        if 'multi_plot_images' in st.session_state:
            st.session_state['multi_plot_images']['multiple_testing_correction_pvalues.png'] = plt_bytes

        st.download_button(
            "📥 Download Multi-Correction Comparison Plot (PNG)",
            data=plt_bytes,
            file_name="multiple_testing_correction_pvalues.png",
            mime="image/png",
            key=f"dl_mt_plot_{key_suffix}"
        )

    with tab_hist:
        fig_phist = generate_pvalue_histogram(p_values_dict, alpha=alpha_val, dpi=130)
        st.pyplot(fig_phist)

        buf_phist = io.BytesIO()
        fig_phist.savefig(buf_phist, format='png', dpi=300, bbox_inches='tight')
        phist_bytes = buf_phist.getvalue()
        plt.close(fig_phist)

        if image_collector:
            image_collector('pairwise_pvalue_distribution_histogram.png', phist_bytes)
        if 'multi_plot_images' in st.session_state:
            st.session_state['multi_plot_images']['pairwise_pvalue_distribution_histogram.png'] = phist_bytes

        st.download_button(
            "📥 Download P-Value Distribution Histogram (PNG)",
            data=phist_bytes,
            file_name="pairwise_pvalue_distribution_histogram.png",
            mime="image/png",
            key=f"dl_phist_plot_{key_suffix}"
        )

    with tab_tbl:
        # Format table for display
        df_disp = df_corr.copy()
        for col in ["Raw p-value", "Bonferroni p_adj", "Holm p_adj", "BH FDR q-value", "BY FDR q-value", "Šidák p_adj"]:
            df_disp[col] = df_disp[col].apply(lambda x: f"< 0.0001" if x < 0.0001 else f"{x:.4f}")

        # Human-readable decision column based on selected method
        if "Bonferroni" in method_view:
            df_disp["Decision"] = df_corr["Sig (Bonferroni)"].apply(lambda x: "✅ Reject H₀ (Significant)" if x else "⚪ Retain H₀")
        elif "Holm" in method_view:
            df_disp["Decision"] = df_corr["Sig (Holm)"].apply(lambda x: "✅ Reject H₀ (Significant)" if x else "⚪ Retain H₀")
        elif "BY" in method_view:
            df_disp["Decision"] = df_corr["Sig (BY FDR)"].apply(lambda x: "✅ Reject H₀ (Significant Discovery)" if x else "⚪ Retain H₀")
        elif "Šidák" in method_view:
            df_disp["Decision"] = df_corr["Sig (Šidák)"].apply(lambda x: "✅ Reject H₀ (Significant)" if x else "⚪ Retain H₀")
        else:
            df_disp["Decision"] = df_corr["Sig (BH FDR)"].apply(lambda x: "✅ Reject H₀ (Significant Discovery)" if x else "⚪ Retain H₀")

        st.dataframe(df_disp, use_container_width=True)

        csv_bytes = df_corr.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📊 Download Multi-Correction Summary Table (CSV)",
            data=csv_bytes,
            file_name="multi_testing_correction_fdr_summary.csv",
            mime="text/csv",
            key=f"dl_mt_table_csv_{key_suffix}"
        )
        st.session_state['multi_testing_summary_df'] = df_corr

