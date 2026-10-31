"""
AI Plot Studio & ChatGPT Code Runner Module.
Provides an interactive bridge for non-coding researchers and professors to:
1. Generate context-aware prompt templates for ChatGPT (including live bioassay data schemas).
2. Execute ChatGPT-generated Python plot code safely in an interactive sandbox.
3. Preview and export publication-ready 300 DPI figures immediately.
"""
import io
import sys
import traceback
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
import scipy.stats as stats
from lifelines import KaplanMeierFitter, CoxPHFitter

from modules.core.time_parser import filter_post_starvation_data
from modules.parser import (
    extract_individual_bee_survival,
    get_default_4_cohort_dataframes,
    clean_cohort_name,
    get_cohort_code,
    get_bee_columns,
    is_bee_col,
    COHORT_PALETTE
)


# Starter Template Code Snippets
TEMPLATE_CODE_SNIPPETS = {
    "1. Kaplan-Meier Survival Curves with Confidence Bands": """# Kaplan-Meier Product-Limit Survival Curves with 95% Greenwood Confidence Bands
fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)
ax.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', zorder=1)

kmf = KaplanMeierFitter()
palette = ['#2563eb', '#0d9488', '#f59e0b', '#dc2626', '#8b5cf6', '#ec4899']

# Group by cohort
cohorts = survival_df['cohort'].unique()
for idx, cohort in enumerate(cohorts):
    sub = survival_df[survival_df['cohort'] == cohort]
    durations = sub['duration']
    events = sub['event']
    color = palette[idx % len(palette)]
    
    kmf.fit(durations, event_observed=events, label=f"{cohort} (n={len(sub)})")
    kmf.plot_survival_function(ax=ax, color=color, ci_show=True, ci_alpha=0.15, linewidth=2.2)

ax.set_title("Kaplan-Meier Survival Curves (Post-2h Starvation)", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Observation Time (Days Post-Starvation)", fontsize=11, fontweight='bold')
ax.set_ylabel("Survival Probability Ŝ(t)", fontsize=11, fontweight='bold')
ax.set_ylim(-0.02, 1.05)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=9.5)
plt.tight_layout()
""",

    "2. Comparative LT50 & LT20 Bar Chart": """# Comparative Lethal Time (LT50 & LT20) Bar Chart
fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
ax.grid(True, linestyle='--', alpha=0.35, color='#e2e8f0', axis='y', zorder=1)

if not summary_df.empty and 'LT50' in summary_df.columns:
    cohort_names = summary_df['CSV File Name'].tolist()
    
    # Extract numeric values
    def parse_days(val):
        try:
            return float(str(val).replace('d', '').replace('Days', '').strip())
        except Exception:
            return np.nan
            
    lt50_vals = [parse_days(v) for v in summary_df['LT50']]
    lt20_vals = [parse_days(v) for v in summary_df.get('LT20', [np.nan]*len(cohort_names))]
    
    x = np.arange(len(cohort_names))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, lt50_vals, width, label='LT50 (Median)', color='#2563eb', edgecolor='black')
    rects2 = ax.bar(x + width/2, lt20_vals, width, label='LT20 (Early Onset)', color='#0d9488', edgecolor='black')
    
    ax.set_ylabel('Lethal Time (Days Post-Starvation)', fontsize=11, fontweight='bold')
    ax.set_title('Comparative Lethal Time (LT50 vs LT20) Across Cohorts', fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(cohort_names, rotation=25, ha='right', fontsize=9.5)
    ax.legend(frameon=True, facecolor='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels on bars
    for bar in rects1:
        h = bar.get_height()
        if np.isfinite(h) and h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.1, f'{int(round(h))}d', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
    for bar in rects2:
        h = bar.get_height()
        if np.isfinite(h) and h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.1, f'{int(round(h))}d', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
    plt.tight_layout()
else:
    ax.text(0.5, 0.5, "Summary metrics table is currently empty.", ha='center', va='center', fontsize=12)
""",

    "3. Daily Sucrose Consumption Dynamics Boxplot": """# Daily Sucrose Consumption Distribution Across Cohorts
fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)
ax.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', axis='y', zorder=1)

if 'mean_consumption' in survival_df.columns and (survival_df['mean_consumption'] > 0).any():
    cohort_list = [c for c in survival_df['cohort'].unique() if pd.notna(c)]
    if sns is not None:
        sns.boxplot(data=survival_df, x='cohort', y='mean_consumption', palette='Set2', ax=ax, width=0.45)
        sns.stripplot(data=survival_df, x='cohort', y='mean_consumption', color='black', size=5, alpha=0.6, jitter=0.15, ax=ax)
    else:
        box_data = [survival_df[survival_df['cohort'] == c]['mean_consumption'].dropna().values for c in cohort_list]
        bp = ax.boxplot(box_data, patch_artist=True)
        ax.set_xticks(range(1, len(cohort_list) + 1))
        ax.set_xticklabels(cohort_list)
        colors = ['#2563eb', '#0d9488', '#f59e0b', '#dc2626', '#8b5cf6']
        for patch, color in zip(bp['boxes'], colors * (len(cohort_list) // len(colors) + 1)):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
    ax.set_title("Mean Daily Sucrose Consumption per Bee (µL/day)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Experimental Cohort", fontsize=11, fontweight='bold')
    ax.set_ylabel("Daily Intake / Bee (µL)", fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=20, ha='right', fontsize=9.5)
    plt.tight_layout()
else:
    ax.text(0.5, 0.5, "Consumption tracking columns (Cons µL) not detected in dataset.", ha='center', va='center', fontsize=11)
""",

    "4. Bee Survival & Mortality Heatmap Grid": """# Individual Bee Observation Status Heatmap Over Time
fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=150)

# Build numerical status grid from observation sheet
bee_cols = [c for c in df.columns if is_bee_col(c)]
if bee_cols and 'Day' in df.columns:
    state_map = {'A': 2, 'B': 1, 'D': 0}
    matrix = []
    days = df['Day'].tolist()
    
    for b in bee_cols:
        row_vals = []
        for d_idx, day_val in enumerate(days):
            raw_s = str(df.at[d_idx, b]).strip().upper()
            row_vals.append(state_map.get(raw_s, 2))
        matrix.append(row_vals)
        
    matrix_arr = np.array(matrix)
    if sns is not None:
        cmap = sns.color_palette(["#ef4444", "#f59e0b", "#10b981"], as_cmap=True)
        sns.heatmap(
            matrix_arr,
            annot=False,
            cmap=cmap,
            yticklabels=bee_cols,
            xticklabels=days,
            cbar=True,
            cbar_kws={'label': 'Bee State: 2 = Alive (A), 1 = Moving (B), 0 = Dead (D)', 'ticks': [0.33, 1.0, 1.66]},
            linewidths=0.5,
            linecolor='white',
            ax=ax
        )
    else:
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(["#ef4444", "#f59e0b", "#10b981"])
        cax = ax.imshow(matrix_arr, aspect='auto', cmap=cmap, interpolation='nearest')
        ax.set_yticks(np.arange(len(bee_cols)))
        ax.set_yticklabels(bee_cols, fontsize=9)
        ax.set_xticks(np.arange(len(days)))
        ax.set_xticklabels(days, rotation=45, ha='right', fontsize=9)
        cbar = fig.colorbar(cax, ax=ax, ticks=[0.33, 1.0, 1.66])
        cbar.ax.set_yticklabels(['Dead (D)', 'Moving (B)', 'Alive (A)'])
    ax.set_title("Individual Bee Longevity Progression Heatmap", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Observation Days (Post-Starvation)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Individual Bee ID", fontsize=11, fontweight='bold')
    plt.tight_layout()
else:
    ax.text(0.5, 0.5, "No individual bee observation columns found in active dataset.", ha='center', va='center')
""",

    "5. Finney Probit Regression Lines Comparison": """# Finney Probit Linear Regression Comparison
fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=150)
ax.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1', zorder=1)

palette = ['#2563eb', '#0d9488', '#f59e0b', '#dc2626', '#8b5cf6', '#ec4899']

if batches:
    for idx, b in enumerate(batches):
        color = b.get('color', palette[idx % len(palette)])
        log_t = b.get('log_time', [])
        probit_y = b.get('probit_y', [])
        intercept = b.get('intercept_a', 0.0)
        slope = b.get('slope_b', 1.0)
        r2 = b.get('r_squared', 0.0)
        name = b.get('name', f"Cohort {idx+1}")
        
        if len(log_t) > 0:
            ax.scatter(log_t, probit_y, color=color, s=45, edgecolor='black', zorder=3)
            x_line = np.linspace(min(log_t) - 0.05, max(log_t) + 0.05, 50)
            y_line = intercept + slope * x_line
            ax.plot(x_line, y_line, color=color, linewidth=2.0, label=f"{name} (R²={r2:.3f})", zorder=2)

    ax.axhline(5.0, color='#64748b', linestyle='--', linewidth=1.4, label='LT50 Probit Reference (5.0)')
    ax.set_xlabel("Log10(Time / Days)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Probit Value (Y)", fontsize=11, fontweight='bold')
    ax.set_title("Finney Probit Regression Lines", fontsize=13, fontweight='bold', pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=9)
    plt.tight_layout()
else:
    ax.text(0.5, 0.5, "No multi-batch data currently loaded.", ha='center', va='center')
"""
}


def _assemble_live_data_context():
    """Builds the comprehensive live data dictionary available for ChatGPT and the code runner."""
    # 1. Observation sheet DataFrame
    active_df = st.session_state.get('df', pd.DataFrame())
    if active_df.empty:
        active_df = pd.DataFrame()

    # 2. Processed cohorts / batches
    uploaded_dict = st.session_state.get('uploaded_file_dict', {})
    batches = st.session_state.get('processed_batches', [])
    batch_dict = {}

    if uploaded_dict:
        for fname, (c_name, c_df) in uploaded_dict.items():
            batch_dict[fname] = filter_post_starvation_data(c_df.copy())
    else:
        def_dict = get_default_4_cohort_dataframes()
        for c_name, c_df in def_dict.items():
            batch_dict[c_name] = c_df

    # 3. Individual Bee Survival DataFrame
    all_bees = []
    if batches:
        for b in batches:
            b_df = b.get('individual_bees')
            if b_df is not None and not b_df.empty:
                b_copy = b_df.copy()
                b_copy['cohort'] = b.get('name', 'Cohort')
                b_copy['cohort_code'] = b.get('code_name', 'Cohort')
                all_bees.append(b_copy)
    elif batch_dict:
        for fname, c_df in batch_dict.items():
            b_bees = extract_individual_bee_survival(c_df, batch_name=fname)
            if not b_bees.empty:
                b_bees['cohort'] = fname
                b_bees['cohort_code'] = get_cohort_code(fname)
                all_bees.append(b_bees)

    survival_df = pd.concat(all_bees, ignore_index=True) if all_bees else pd.DataFrame()

    # 4. Summary metrics table
    summary_rows = []
    if batches:
        for b in batches:
            lt50_val = b.get('lt50', np.nan)
            lt20_val = b.get('lt20', np.nan)
            summary_rows.append({
                "CSV File Name": b.get('name', 'Cohort'),
                "Sample Size (N)": b.get('num_bees', 10),
                "Slope (b)": round(b.get('slope_b', 0.0), 4),
                "Intercept (a)": round(b.get('intercept_a', 0.0), 4),
                "R² Score": round(b.get('r_squared', 0.0), 4),
                "LT20": f"{int(round(lt20_val))}d" if np.isfinite(lt20_val) and lt20_val > 0 else "N/A",
                "LT50": f"{int(round(lt50_val))}d" if np.isfinite(lt50_val) and lt50_val > 0 else "N/A",
                "Final Mortality (%)": f"{b.get('mortality_pct', [0])[-1]:.1f}%" if len(b.get('mortality_pct', [])) > 0 else "N/A"
            })
    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()

    return {
        'df': active_df,
        'survival_df': survival_df,
        'summary_df': summary_df,
        'batch_dict': batch_dict,
        'batches': batches
    }


def _generate_chatgpt_prompt(user_query: str, data_ctx: dict) -> str:
    """Generates a complete, structured ChatGPT prompt with live schema and bioassay protocol rules."""
    df = data_ctx['df']
    survival_df = data_ctx['survival_df']
    summary_df = data_ctx['summary_df']
    batch_dict = data_ctx['batch_dict']

    # Summarize df
    df_cols = list(df.columns) if not df.empty else []
    df_preview = df.head(3).to_dict(orient='records') if not df.empty else []

    # Summarize survival_df
    surv_cols = list(survival_df.columns) if not survival_df.empty else []
    cohorts_list = list(survival_df['cohort'].unique()) if (not survival_df.empty and 'cohort' in survival_df.columns) else list(batch_dict.keys())

    prompt = f"""You are an expert biostatistician and scientific data visualization specialist in Python (matplotlib, seaborn, lifelines).
I am analyzing honeybee bioassay mortality and longevity datasets. Write clean, complete Python code to generate a publication-ready plot according to my request.

---
### 🔬 MY PLOT REQUEST:
"{user_query.strip()}"

---
### 📊 AVAILABLE IN-MEMORY PYTHON VARIABLES:
Your code will be executed in a Python environment where the following variables are ALREADY loaded and ready to use:

1. `survival_df` (pandas.DataFrame, shape {survival_df.shape}):
   - Subject-level individual bee tracking table for survival analysis.
   - Columns: {surv_cols}
   - Distinct Cohorts: {cohorts_list}
   - Key Fields:
     * `duration`: Float elapsed days until event/censoring.
     * `event`: Integer (1 = Dead, 0 = Censored / Alive).
     * `cohort`: Categorical display name of experimental group.
     * `cohort_code`: Standard code name (e.g. 'Sucrose 2 µL', '5% Acetone 2 µL').
     * `mean_consumption`: Average daily sucrose consumption in µL (if tracked).
     * `temp`: Mean observed temperature in °C.

2. `df` (pandas.DataFrame, shape {df.shape}):
   - Raw observation grid.
   - Columns: {df_cols}
   - Sample row: {df_preview[:1]}

3. `summary_df` (pandas.DataFrame, shape {summary_df.shape}):
   - Summary statistics table.
   - Columns: {list(summary_df.columns) if not summary_df.empty else []}

4. `batches` (list of dicts):
   - Parsed cohort metrics containing: 'name', 'color', 'marker', 'log_time', 'probit_y', 'slope_b', 'intercept_a', 'r_squared', 'lt50', 'mortality_pct', 'survival_pct'.

5. Already imported libraries:
   `import matplotlib.pyplot as plt`
   `import seaborn as sns`
   `import pandas as pd`
   `import numpy as np`
   `import scipy.stats as stats`
   `from lifelines import KaplanMeierFitter, CoxPHFitter`

---
### ⚠️ STRICT BIOASSAY PROTOCOL & CODING RULES:
1. **Post-2h Starvation Rule**: Bees undergo a 2-hour starvation period prior to treatment exposure. Day 0 (t=0) is pre-starvation baseline and must NEVER be plotted or included in calculations. All active observations start on Day 1 (t=1.0).
2. **Output Variable**: Create the figure using `fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)` and ensure the figure object is stored in the variable `fig`.
3. **NO `plt.show()`**: Do NOT call `plt.show()`, as this is rendered inside Streamlit using `fig`.
4. **Publication Quality**: Use clean borders (`ax.spines['top'].set_visible(False)`, `ax.spines['right'].set_visible(False)`), clear bold axis labels, and high-contrast color palette.
5. Provide ONLY pure, executable Python code in a single code block.
"""
    return prompt


def render_ai_plots_tab():
    """Renders the AI Plot Studio tab with prompt generator and Python code runner."""
    st.header("🤖 AI Plot Studio (ChatGPT Code Runner)")
    st.markdown("Create custom scientific figures without writing code from scratch. Describe what you want, generate an optimized query prompt for **ChatGPT / Claude**, and paste the Python code back here to run and download high-resolution plots.")

    st.info("🕒 **Bioassay Context Active**: Active datasets, individual bee durations, Finney Probit parameters, and post-2h starvation guidelines are automatically formatted into your prompt.")

    data_ctx = _assemble_live_data_context()

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        n_cohorts = len(data_ctx['batch_dict'])
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{n_cohorts}</div><div class="kpi-label">Active Cohorts</div></div>', unsafe_allow_html=True)
    with col_info2:
        n_bees = len(data_ctx['survival_df']) if not data_ctx['survival_df'].empty else 0
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{n_bees}</div><div class="kpi-label">Total Bees in survival_df</div></div>', unsafe_allow_html=True)
    with col_info3:
        n_days = len(data_ctx['df']) if not data_ctx['df'].empty else 0
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{n_days}</div><div class="kpi-label">Observation Days in df</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab_prompt, tab_runner = st.tabs([
        "📝 Step 1: Generate ChatGPT Prompt Template",
        "▶️ Step 2: Code Runner & Interactive Plot Canvas"
    ])

    # TAB 1: PROMPT GENERATOR
    with tab_prompt:
        st.subheader("1. Build Your Custom ChatGPT Query Prompt")
        st.markdown("Select an idea or type what you would like to visualize, then copy the prompt template below and paste it into ChatGPT (GPT-4o / Claude).")

        PRESET_IDEAS = [
            "Bar chart comparing LT50 values across cohorts with error bars and p-value markers",
            "Kaplan-Meier survival step curves with 95% Greenwood confidence bands and number-at-risk table",
            "Grouped boxplot of mean daily sucrose consumption by treatment cohort with jittered points",
            "Individual bee survival state progression heatmap (Alive, Antennae moving, Dead) over observation days",
            "Comparative Finney Probit linear regression lines with log-time points and R² labels",
            "Daily instantaneous mortality hazard rate curve over observation days",
            "Custom Query (Type below)..."
        ]

        sel_preset = st.selectbox("💡 Quick-Pick Plot Idea or Inspiration:", PRESET_IDEAS, index=0)

        if sel_preset == "Custom Query (Type below)...":
            user_query = st.text_area(
                "Enter your specific plot request in plain English:",
                value="Create a publication-quality stacked bar chart showing the percentage of bees Alive (A), Moving (B), and Dead (D) on each observation day for each cohort.",
                height=90
            )
        else:
            user_query = sel_preset

        generated_prompt = _generate_chatgpt_prompt(user_query, data_ctx)

        st.markdown("#### 📋 Copy This Prompt to ChatGPT:")
        st.code(generated_prompt, language="markdown")
        st.caption("👉 **Next Step**: Paste the prompt into ChatGPT. When ChatGPT provides the Python code snippet, copy and paste it into **Step 2 (Code Runner & Interactive Plot Canvas)** above!")

    # TAB 2: CODE RUNNER & INTERACTIVE CANVAS
    with tab_runner:
        st.subheader("2. Python Plot Sandbox & Code Runner")
        st.markdown("Paste code from ChatGPT below, or choose one of the built-in starter templates to run immediately:")

        c_tmpl, c_clear = st.columns([3, 1])
        with c_tmpl:
            tmpl_choice = st.selectbox(
                "✨ Load Ready-to-Run Template Snippet:",
                list(TEMPLATE_CODE_SNIPPETS.keys()),
                index=0,
                key="ai_plot_template_selector"
            )
        with c_clear:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🔄 Reset to Template", use_container_width=True):
                st.session_state['user_plot_code'] = TEMPLATE_CODE_SNIPPETS[tmpl_choice]
                st.rerun()

        if 'user_plot_code' not in st.session_state or st.session_state.get('last_tmpl') != tmpl_choice:
            st.session_state['user_plot_code'] = TEMPLATE_CODE_SNIPPETS[tmpl_choice]
            st.session_state['last_tmpl'] = tmpl_choice

        code_input = st.text_area(
            "Python Plot Code (Matplotlib / Seaborn / Lifelines):",
            value=st.session_state['user_plot_code'],
            height=300,
            key="ai_code_editor"
        )
        st.session_state['user_plot_code'] = code_input

        btn_run = st.button("▶️ Run Python Plot Code", type="primary", use_container_width=True)

        # Code Execution & Rendering Engine
        if btn_run or code_input:
            exec_scope = {
                'plt': plt,
                'sns': sns,
                'pd': pd,
                'np': np,
                'stats': stats,
                'KaplanMeierFitter': KaplanMeierFitter,
                'CoxPHFitter': CoxPHFitter,
                'df': data_ctx['df'],
                'survival_df': data_ctx['survival_df'],
                'summary_df': data_ctx['summary_df'],
                'batch_dict': data_ctx['batch_dict'],
                'batches': data_ctx['batches'],
                '__builtins__': __builtins__
            }

            try:
                # Clean up previous matplotlib plots
                plt.close('all')

                # Execute Python snippet
                exec(code_input, exec_scope)

                # Retrieve generated figure
                fig = exec_scope.get('fig') or plt.gcf()

                if fig and len(fig.get_axes()) > 0:
                    st.success("✅ **Plot Successfully Generated!**")
                    st.pyplot(fig)

                    # Export high-res 300 DPI version
                    buf_png = io.BytesIO()
                    fig.savefig(buf_png, format="png", dpi=300, bbox_inches='tight')
                    buf_svg = io.BytesIO()
                    fig.savefig(buf_svg, format="svg", bbox_inches='tight')

                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        st.download_button(
                            "📥 Download Publication Plot (PNG @ 300 DPI)",
                            data=buf_png.getvalue(),
                            file_name="chatgpt_custom_plot_300dpi.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    with col_dl2:
                        st.download_button(
                            "📥 Download Vector Plot (SVG)",
                            data=buf_svg.getvalue(),
                            file_name="chatgpt_custom_plot.svg",
                            mime="image/svg+xml",
                            use_container_width=True
                        )
                else:
                    st.warning("⚠️ Code executed successfully, but no matplotlib axes or plot was produced. Make sure your code creates a plot, e.g. `fig, ax = plt.subplots()`.")

            except Exception as e:
                err_msg = traceback.format_exc()
                st.error(f"❌ **Code Execution Error**: `{type(e).__name__}: {str(e)}`")
                with st.expander("🔍 View Error Traceback & ChatGPT Correction Advice", expanded=True):
                    st.code(err_msg, language="python")
                    st.markdown("""
                    **💡 How to fix:**
                    - If you see `KeyError: '...'`, verify the column names in `survival_df` or `df`.
                    - You can copy this error message and paste it back to ChatGPT: *"I got this error when running the code: [paste error here]. Please fix the code for me."*
                    """)

