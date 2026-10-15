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


def render_kaplan_meier_math_card():
    """Renders the detailed mathematical methodology explanation card for the Kaplan-Meier estimator."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(248,250,252,0.95), rgba(241,245,249,0.95)); border: 1px solid #cbd5e1; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
        <h4 style="margin-top: 0; color: #0f172a; display: flex; align-items: center; gap: 8px;">
            📐 Kaplan–Meier Product-Limit Estimator
        </h4>
        <p style="font-size: 0.9rem; color: #475569; margin-bottom: 12px;">
            The <b>Kaplan–Meier estimator</b> calculates survival by multiplying the conditional survival probabilities at each successive death time.
        </p>
        <div style="background: white; border-radius: 8px; padding: 12px; border-left: 4px solid #0284c7; margin-bottom: 14px;">
            <b style="color: #0369a1;">The Formula:</b>
            <div style="margin: 8px 0; font-size: 1.05rem; text-align: center;">
                $$\\hat{S}(t) = \\prod_{t_i \\le t} \\left(1 - \\frac{d_i}{n_i}\\right)$$
            </div>
            <span style="font-size: 0.84rem; color: #475569; display: block; line-height: 1.5;">
                <b>Where:</b><br>
                • $\\hat{S}(t)$ = estimated probability of surviving beyond observation time $t$.<br>
                • $d_i$ = number of observed deaths occurring at time $t_i$.<br>
                • $n_i$ = number of bees at risk immediately before that time.<br>
                • $\\prod$ means that the conditional probabilities are multiplied together over successive days.
            </span>
        </div>
        <div style="background: white; border-radius: 8px; padding: 12px; border-left: 4px solid #10b981; margin-bottom: 14px;">
            <b style="color: #047857;">Step-by-Step Bioassay Example:</b>
            <div style="font-size: 0.85rem; color: #334155; margin-top: 6px; line-height: 1.7;">
                <b>After Day 1:</b>
                $$S(1) = \\frac{9}{10} = 0.90$$
                <b>After Day 2:</b>
                $$S(2) = 0.90 \\times \\frac{8}{9} = 0.80$$
                <b>After Day 3:</b>
                $$S(3) = 0.80 \\times \\frac{7}{8} = 0.70$$
                <span style="color: #047857; font-weight: bold;">70% estimated survival after Day 3</span><br>
                <em>Seven of the original ten bees remain alive at the end of Day 3 in this example.</em>
            </div>
        </div>
        <div style="background: white; border-radius: 8px; padding: 12px; border-left: 4px solid #f59e0b; margin-bottom: 14px;">
            <b style="color: #b45309;">What is Censoring, and Why is it Important?</b>
            <p style="font-size: 0.83rem; color: #475569; margin: 6px 0 0 0; line-height: 1.45;">
                In pesticide bioassays, bees that remain alive at the conclusion of the test period (or are removed alive) are <b>right-censored</b> ($event = 0$). Censoring is vital because simply dropping censored bees biases survival downward, whereas treating them as dead biases survival upward. The Kaplan–Meier estimator keeps censored individuals in the risk denominator $n_i$ up to their censoring day, yielding mathematically unbiased survival probabilities.
            </p>
        </div>
        <div style="font-size: 0.82rem; color: #64748b;">
            <b>Greenwood's 95% Confidence Band (Survival Uncertainty):</b>
            $$\\widehat{\\text{SE}}(\\hat{S}(t)) = \\hat{S}(t) \\sqrt{\\sum_{t_i \\le t} \\frac{d_i}{n_i (n_i - d_i)}}$$
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_methodology_comparison_card():
    """Renders the comprehensive 4-way methodological distinction table from the bioassay guide."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(248,250,252,0.95), rgba(241,245,249,0.95)); border: 1px solid #cbd5e1; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
        <h4 style="margin-top: 0; color: #0f172a; display: flex; align-items: center; gap: 8px;">
            📊 Kaplan–Meier vs. Log-Rank vs. Cox Regression vs. Probit Analysis
        </h4>
        <p style="font-size: 0.88rem; color: #475569; margin-bottom: 14px;">
            These core statistical methods are closely related, but they answer fundamentally distinct scientific questions in toxicological bioassays:
        </p>
        <div style="overflow-x: auto;">
            <table style="width: 100%; font-size: 0.84rem; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <thead>
                    <tr style="background: #f1f5f9; text-align: left; color: #1e293b; border-bottom: 2px solid #cbd5e1;">
                        <th style="padding: 10px 12px;">Method</th>
                        <th style="padding: 10px 12px;">Main Question Answered</th>
                        <th style="padding: 10px 12px;">Typical Output</th>
                        <th style="padding: 10px 12px;">Pesticide Bioassay Role</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid #e2e8f0;">
                        <td style="padding: 10px 12px; font-weight: bold; color: #0284c7;">Kaplan–Meier</td>
                        <td style="padding: 10px 12px;">How does survival change over time?</td>
                        <td style="padding: 10px 12px;">Step survival curve $\\hat{S}(t)$ & median survival ($LT_{50}$)</td>
                        <td style="padding: 10px 12px;">Non-parametric cumulative survival trajectory with censoring.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #e2e8f0; background: #fafafa;">
                        <td style="padding: 10px 12px; font-weight: bold; color: #059669;">Log-Rank Test</td>
                        <td style="padding: 10px 12px;">Do the survival curves differ statistically?</td>
                        <td style="padding: 10px 12px;">Test statistic ($\\chi^2$), $df$, and $p$-value</td>
                        <td style="padding: 10px 12px;">Hypothesis test evaluating if treatment cohorts diverge significantly.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #e2e8f0;">
                        <td style="padding: 10px 12px; font-weight: bold; color: #7c3aed;">Cox Regression</td>
                        <td style="padding: 10px 12px;">How is treatment associated with the instantaneous risk of death?</td>
                        <td style="padding: 10px 12px;">Hazard ratio (HR), 95% CI, Wald $p$-value</td>
                        <td style="padding: 10px 12px;">Semi-parametric model quantifying relative risk & environmental covariates.</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px 12px; font-weight: bold; color: #dc2626;">Probit Analysis</td>
                        <td style="padding: 10px 12px;">What dose or time is associated with a specified mortality probability?</td>
                        <td style="padding: 10px 12px;">$LD_{50}$ / $LT_{50}$, slope ($b$), intercept ($a$), $R^2$</td>
                        <td style="padding: 10px 12px;">Standard linear tolerance distribution for toxicological thresholds.</td>
                    </tr>
                </tbody>
            </table>
        </div>
        <div style="background: #eff6ff; border-left: 4px solid #3b82f6; border-radius: 6px; padding: 10px 14px; margin-top: 14px; font-size: 0.83rem; color: #1e40af;">
            💡 <b>Important Bioassay Distinction:</b> The curve alone does not establish statistical significance or prove that acetone/pesticide caused the difference. For your pesticide bioassays, Kaplan–Meier and log-rank analysis are useful for comparing mortality timing, while Cox regression estimates the treatment-associated hazard ratio. Note that <b>Kaplan–Meier does not directly estimate an LD50 dose</b>—Probit analysis is used for that purpose.
        </div>
    </div>
    """, unsafe_allow_html=True)

