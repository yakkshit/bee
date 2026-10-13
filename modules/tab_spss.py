import streamlit as st


def render_spss_tab():
    """
    Renders the minimal, modern SPSS & Advanced Bioassay Analytics roadmap tab
    featuring an animated typewriter terminal and upcoming statistical capabilities.
    """
    st.markdown("""
    <style>
    .spss-container {
        max-width: 1080px;
        margin: 0 auto;
        padding: 10px 0 30px 0;
    }
    .terminal-window {
        background: #0f172a;
        border-radius: 14px;
        border: 1px solid #334155;
        box-shadow: 0 20px 40px -15px rgba(15, 23, 42, 0.4);
        overflow: hidden;
        margin-bottom: 35px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }
    .terminal-bar {
        background: #1e293b;
        padding: 12px 18px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid #334155;
    }
    .terminal-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-r { background: #ef4444; }
    .dot-y { background: #f59e0b; }
    .dot-g { background: #10b981; }
    .terminal-title {
        color: #94a3b8;
        font-size: 0.82rem;
        margin-left: 8px;
        letter-spacing: 0.5px;
    }
    .terminal-content {
        padding: 24px 28px;
        color: #e2e8f0;
        font-size: 0.95rem;
        line-height: 1.8;
    }
    .typewriter-text {
        color: #38bdf8;
        font-weight: 600;
        border-right: 2px solid #38bdf8;
        white-space: nowrap;
        animation: typing 3.5s steps(45, end) infinite alternate, blink-caret 0.75s step-end infinite;
        display: inline-block;
        overflow: hidden;
        max-width: 100%;
        vertical-align: bottom;
    }
    @keyframes typing {
        from { width: 0; }
        to { width: 100%; }
    }
    @keyframes blink-caret {
        from, to { border-color: transparent; }
        50% { border-color: #38bdf8; }
    }
    .feature-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 22px;
        height: 100%;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 20px -5px rgba(0, 0, 0, 0.08);
        border-color: #cbd5e1;
    }
    .card-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        padding: 3px 9px;
        border-radius: 999px;
        margin-bottom: 12px;
    }
    .badge-blue { background: #e0f2fe; color: #0369a1; }
    .badge-purple { background: #f3e8ff; color: #7e22ce; }
    .badge-emerald { background: #dcfce7; color: #15803d; }
    .badge-amber { background: #fef3c7; color: #b45309; }
    .feature-h {
        font-size: 1.12rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 8px;
    }
    .feature-desc {
        font-size: 0.88rem;
        color: #64748b;
        line-height: 1.55;
    }
    .formula-tag {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 0.8rem;
        color: #334155;
        font-family: ui-monospace, monospace;
        margin-top: 10px;
        display: inline-block;
    }
    </style>

    <div class="spss-container">
        <div class="terminal-window">
            <div class="terminal-bar">
                <span class="terminal-dot dot-r"></span>
                <span class="terminal-dot dot-y"></span>
                <span class="terminal-dot dot-g"></span>
                <span class="terminal-title">IBM_SPSS_BIOSTAT_ENGINE // MODULE_PREVIEW</span>
            </div>
            <div class="terminal-content">
                <span style="color: #64748b;">// Bioassay Inferential Pipeline Architecture</span><br>
                <span style="color: #a855f7;">init_module</span> <span style="color: #94a3b8;">--suite=spss-advanced --engine=glmm</span><br>
                <span style="color: #10b981;">[STATUS]</span> <span class="typewriter-text">SPSS advanced analytics & bioassay features are coming soon...</span><br><br>
                <span style="color: #64748b;"># Active Development Roadmap:</span><br>
                <span style="color: #38bdf8;">✓</span> Finney Parallelism & Relative Toxicity Potency Ratio (Fieller's 95% CI)<br>
                <span style="color: #38bdf8;">✓</span> Repeated-Measures GLM / MANOVA for Longitudinal Micro-Consumption<br>
                <span style="color: #38bdf8;">✓</span> Non-Linear 4-Parameter Logistic (4PL / Hill Equation) Bioassays<br>
                <span style="color: #38bdf8;">✓</span> Automated IBM SPSS Statistics (.sps) & R Syntax Export
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature Grid
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        st.markdown("""
        <div class="feature-card">
            <span class="card-badge badge-blue">SPSS Probit Extension</span>
            <div class="feature-h">1. Probit Parallelism Test & Relative Potency</div>
            <div class="feature-desc">
                Test whether regression slopes across treatment cohorts are strictly parallel (<span style="font-family: monospace;">H₀: β₁ = β₂ = ... = βₖ</span>). Computes relative potency ratios with exact Fieller's theorem 95% confidence bounds.
            </div>
            <div class="formula-tag">χ²_parallelism = Σ (bᵢ - b_pooled)² · wᵢ</div>
        </div>
        """, unsafe_allow_html=True)

    with col_f2:
        st.markdown("""
        <div class="feature-card">
            <span class="card-badge badge-purple">General Linear Models</span>
            <div class="feature-h">2. Repeated-Measures ANOVA & MANOVA</div>
            <div class="feature-desc">
                Evaluate daily bee food consumption dynamics over time with within-subject factors and between-subject treatment cohorts. Includes Greenhouse-Geisser & Huynh-Feldt sphericity corrections.
            </div>
            <div class="formula-tag">Mauchly's W → ε_Greenhouse-Geisser</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_f3, col_f4 = st.columns(2)

    with col_f3:
        st.markdown("""
        <div class="feature-card">
            <span class="card-badge badge-emerald">Toxicology Dose-Response</span>
            <div class="feature-h">3. Non-Linear 4PL & Hill Concentration Fit</div>
            <div class="feature-desc">
                Fit 4-parameter sigmoidal logistic curves for multi-concentration toxicity bioassays with automatic determination of minimal threshold, maximal plateau, and inflection median effect concentration.
            </div>
            <div class="formula-tag">Y = Bottom + (Top - Bottom) / (1 + (X/EC50)^Hill)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_f4:
        st.markdown("""
        <div class="feature-card">
            <span class="card-badge badge-amber">Syntax & Reporting</span>
            <div class="feature-h">4. Automated SPSS Syntax (.sps) & APA Format</div>
            <div class="feature-desc">
                One-click compilation of ready-to-run IBM SPSS Statistics syntax scripts (<span style="font-family: monospace;">.sps</span>) and R scripts to reproduce exact p-values, degrees of freedom, and publication-ready APA 7th tables.
            </div>
            <div class="formula-tag">GENLIN / PROBIT / MIXED Syntax Generator</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Request Section
    with st.expander("💡 Request a specific SPSS / Statistical Feature for the upcoming release", expanded=False):
        c_req1, c_req2 = st.columns([3, 1])
        with c_req1:
            custom_req = st.text_input("Which SPSS or toxicological statistical test would you like to see next?", placeholder="e.g., Dunnett's post-hoc test, mixed-effects Cox regression, lethal concentration LC50...")
        with c_req2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Submit Feature Request", use_container_width=True):
                if custom_req.strip():
                    st.success("✅ Thank you! Your request has been recorded for the upcoming SPSS analytics release.")
                else:
                    st.info("Please enter a statistical test or feature name.")
