"""
Mathematical Methodology Educational Cards for Survival Analysis.
Provides formatted LaTeX equations and clinical/biological context
for Log-Rank and Cox Proportional Hazards tests.
"""
import streamlit as st


def render_logrank_math_card():
    """Renders the mathematical methodology explanation card for the Log-Rank test."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(248,250,252,0.95), rgba(241,245,249,0.95)); border: 1px solid #cbd5e1; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
        <h4 style="margin-top: 0; color: #0f172a; display: flex; align-items: center; gap: 8px;">
            📐 Log-Rank Mathematical Framework
        </h4>
        <p style="font-size: 0.9rem; color: #475569; margin-bottom: 12px;">
            The <b>Mantel-Cox Log-Rank Test</b> is a non-parametric rank test comparing the survival distributions of two or more independent cohorts across the entire observation period under right-censored conditions.
        </p>
        <div style="background: white; border-radius: 8px; padding: 12px; border-left: 4px solid #0284c7; margin-bottom: 14px;">
            <b style="color: #0369a1;">Mantel-Cox Test Statistic:</b>
            <div style="margin: 8px 0; font-size: 0.95rem;">
                $$\\chi^2 = \\frac{\\left(\\sum_{j=1}^K (O_{1j} - E_{1j})\\right)^2}{\\sum_{j=1}^K V_{1j}}$$
            </div>
            <span style="font-size: 0.82rem; color: #64748b;">
                Where $O_{1j}$ is observed deaths in group 1 at time $t_j$, $E_{1j} = d_j \\frac{n_{1j}}{n_j}$ is expected deaths under null hypothesis $H_0$, and $V_{1j} = \\frac{n_{1j} n_{2j} d_j (n_j - d_j)}{n_j^2 (n_j - 1)}$ is hypergeometric variance.
            </span>
        </div>
        <div style="background: white; border-radius: 8px; padding: 12px; border-left: 4px solid #10b981; margin-bottom: 14px;">
            <b style="color: #047857;">Kaplan-Meier Product-Limit Estimator:</b>
            <div style="margin: 8px 0; font-size: 0.95rem;">
                $$\\hat{S}(t) = \\prod_{t_i \\le t} \\left(1 - \\frac{d_i}{n_i}\\right)$$
            </div>
            <span style="font-size: 0.82rem; color: #64748b;">
                Greenwood's 95% Confidence Band (Survival Uncertainty Ribbon):
                $$\\widehat{\\text{Var}}(\\hat{S}(t)) = \\hat{S}(t)^2 \\sum_{t_i \\le t} \\frac{d_i}{n_i (n_i - d_i)}$$
            </span>
        </div>
        <div style="font-size: 0.85rem; color: #475569;">
            <b>Assumptions:</b>
            <ul style="padding-left: 18px; margin: 6px 0 0 0;">
                <li>Censoring is non-informative (independent of event risk).</li>
                <li>Probabilities of survival are identical for subjects recruited early vs late.</li>
                <li>Events occur independently across individuals.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_cox_math_card():
    """Renders the mathematical methodology explanation card for Cox Proportional Hazards."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(248,250,252,0.95), rgba(241,245,249,0.95)); border: 1px solid #cbd5e1; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
        <h4 style="margin-top: 0; color: #0f172a; display: flex; align-items: center; gap: 8px;">
            📐 Cox Proportional Hazards Math & Uncertainty
        </h4>
        <p style="font-size: 0.9rem; color: #475569; margin-bottom: 12px;">
            The <b>Cox Proportional Hazards Model</b> is a semi-parametric regression model that links multiple explanatory covariates directly to the instantaneous rate of mortality (hazard function).
        </p>
        <div style="background: white; border-radius: 8px; padding: 12px; border-left: 4px solid #ef4444; margin-bottom: 14px;">
            <b style="color: #b91c1c;">Hazard Function Formulation:</b>
            <div style="margin: 8px 0; font-size: 0.95rem;">
                $$h(t \\mid X) = h_0(t) \\exp\\left(\\sum_{j=1}^p \\beta_j X_j\\right)$$
            </div>
            <span style="font-size: 0.82rem; color: #64748b;">
                Where $h_0(t)$ is the non-parametric baseline hazard over elapsed days, and $\\beta_j$ is the partial maximum likelihood regression coefficient.
            </span>
        </div>
        <div style="background: white; border-radius: 8px; padding: 12px; border-left: 4px solid #8b5cf6; margin-bottom: 14px;">
            <b style="color: #6d28d9;">Hazard Ratio (HR) & 95% Confidence Intervals:</b>
            <div style="margin: 8px 0; font-size: 0.95rem;">
                $$\\text{HR}_j = \\exp(\\beta_j), \\quad 95\\% \\text{ CI} = \\left[\\exp(\\beta_j - 1.96 \\cdot \\text{SE}), \\; \\exp(\\beta_j + 1.96 \\cdot \\text{SE})\\right]$$
            </div>
            <span style="font-size: 0.82rem; color: #64748b;">
                The 95% Confidence Interval quantifies <b>parameter estimation uncertainty</b>. If the interval excludes 1.0, the covariate significantly modifies survival rate ($p < 0.05$).
            </span>
        </div>
        <div style="font-size: 0.85rem; color: #475569;">
            <b>Discrimination & Quality Metrics:</b>
            <ul style="padding-left: 18px; margin: 6px 0 0 0;">
                <li><b>C-Index (Harrell's C):</b> Concordance probability ($0.5 = \\text{chance}$, $1.0 = \\text{perfect ranking}$).</li>
                <li><b>Partial AIC:</b> Akaike Information Criterion assessing goodness-of-fit with parsimony penalty.</li>
                <li><b>Proportionality:</b> Assumes hazard ratios between groups remain constant over time.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
