import os
os.environ['MPLCONFIGDIR'] = os.path.abspath('./.matplotlib_cache')
os.makedirs('./.matplotlib_cache', exist_ok=True)
os.makedirs('./plots', exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch
from scipy.optimize import curve_fit
from lifelines import KaplanMeierFitter
import shutil

# Publication-grade typography and aesthetic parameters matching the paper
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#111111'
plt.rcParams['axes.linewidth'] = 1.6
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['xtick.major.size'] = 5.5
plt.rcParams['xtick.major.width'] = 1.6
plt.rcParams['ytick.major.size'] = 5.5
plt.rcParams['ytick.major.width'] = 1.6
plt.rcParams['font.size'] = 11

# 1. Load and parse dataset
df = pd.read_csv('observation_individual_bees.csv')
days = [c for c in df.columns if c.startswith('d')] # ['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7']
num_days = len(days) # 8 days (0 to 7)

bee_df = df[df['bee_id'].apply(lambda x: str(x).isdigit())].copy()
bee_df['bee_id'] = bee_df['bee_id'].astype(int)
num_bees = len(bee_df)

temp_row = df[df['bee_id'].str.lower() == 'temp'].iloc[0]
hum_row = df[df['bee_id'].str.lower() == 'humidity'].iloc[0]

temps = [float(temp_row[d]) for d in days]
humidities = [float(hum_row[d]) for d in days]

# Determine individual durations, events, and states matrix
durations = []
events = []
states_matrix = []

for _, row in bee_df.iterrows():
    b_states = [row[d] for d in days]
    states_matrix.append(b_states)
    if 'D' in b_states:
        durations.append(b_states.index('D'))
        events.append(1)
    else:
        durations.append(num_days - 1)
        events.append(0)

durations = np.array(durations)
events = np.array(events)

# Daily counts and proportions
day_indices = np.arange(num_days) # 0, 1, 2, ..., 7
alive_counts = []
dead_counts = []

for d_idx in range(num_days):
    n_dead = sum(1 for states in states_matrix if states[d_idx] == 'D')
    alive_counts.append(num_bees - n_dead)
    dead_counts.append(n_dead)

alive_counts = np.array(alive_counts)
dead_counts = np.array(dead_counts)
survival_pct = alive_counts / num_bees * 100.0
mortality_pct = dead_counts / num_bees * 100.0

# Binomial Standard Error for proportions: SE = sqrt(p * (1 - p) / n) * 100
mortality_se = np.sqrt((mortality_pct / 100.0) * (1.0 - mortality_pct / 100.0) / num_bees) * 100.0
survival_se = np.sqrt((survival_pct / 100.0) * (1.0 - survival_pct / 100.0) / num_bees) * 100.0

# -------------------------------------------------------------
# FIT STATISTICAL MODELS
# -------------------------------------------------------------
# Model 1: Sigmoidal / Logistic Mortality Function
def logistic_mort(t, L, k, t0):
    return L / (1.0 + np.exp(-k * (t - t0)))

popt_mort, _ = curve_fit(
    logistic_mort, day_indices, mortality_pct,
    p0=[60.0, 0.5, 5.5],
    bounds=([20.0, 0.01, 0.0], [100.0, 5.0, 15.0]),
    maxfev=10000
)

L_fit, k_fit, t0_fit = popt_mort
residuals_mort = mortality_pct - logistic_mort(day_indices, *popt_mort)
ss_res_mort = np.sum(residuals_mort**2)
ss_tot_mort = np.sum((mortality_pct - np.mean(mortality_pct))**2)
r2_mort = 1.0 - (ss_res_mort / ss_tot_mort) if ss_tot_mort != 0 else 1.0

# Model 2: Parametric Survival Decay Model S(t) = S0 * exp(-a * t^b)
def survival_decay(t, S0, a, b):
    return S0 * np.exp(-a * (t**b))

popt_surv, _ = curve_fit(
    survival_decay, day_indices, survival_pct,
    p0=[78.57, 0.001, 3.0],
    bounds=([50.0, 1e-6, 0.1], [100.0, 2.0, 10.0]),
    maxfev=10000
)

S0_surv, a_surv, b_surv = popt_surv
residuals_surv = survival_pct - survival_decay(day_indices, *popt_surv)
ss_res_surv = np.sum(residuals_surv**2)
ss_tot_surv = np.sum((survival_pct - np.mean(survival_pct))**2)
r2_surv = 1.0 - (ss_res_surv / ss_tot_surv) if ss_tot_surv != 0 else 1.0

# Model 3: Kaplan-Meier Non-Parametric Survival Model
kmf = KaplanMeierFitter()
kmf.fit(durations, event_observed=events, timeline=np.linspace(0, 7, 140))

t_dense = np.linspace(0, 7.5, 250)
fit_mort_curve = logistic_mort(t_dense, *popt_mort)
fit_surv_curve = survival_decay(t_dense, *popt_surv)

# List of filenames to save
output_files = []

# ==============================================================================
# 1. STANDALONE FIGURE: Mortality Curve (Paper Style)
# ==============================================================================
fig, ax = plt.subplots(figsize=(6.8, 5.4), dpi=300)
fit_se = np.std(residuals_mort)

ax.fill_between(t_dense, np.clip(fit_mort_curve - 1.96*fit_se, 0, 100), np.clip(fit_mort_curve + 1.96*fit_se, 0, 100),
                color='#e60000', alpha=0.12, label='95% Confidence Band')
ax.plot(t_dense, fit_mort_curve, color='#e60000', linewidth=2.4, label='Logistic Model Fit')
ax.errorbar(day_indices, mortality_pct, yerr=mortality_se, fmt='o', color='#e60000',
            ecolor='#e60000', elinewidth=1.6, capsize=4, capthick=1.6, markersize=7,
            markerfacecolor='#e60000', markeredgecolor='#b30000', label='Observed (Mean ± SE)')

ax.axhline(50, color='#333333', linestyle=':', linewidth=1.2)
eq_text = f"$Y = \\frac{{{L_fit:.1f}}}{{1 + e^{{-{k_fit:.2f}(X - {t0_fit:.1f})}}}}$\n$R^2 = {r2_mort:.4f}$"
ax.text(0.06, 0.90, eq_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#e0e0e0', alpha=0.85))

ax.set_xlabel('Observation Time / Days', fontsize=13, fontweight='bold', labelpad=7)
ax.set_ylabel('Mortality / %', fontsize=13, fontweight='bold', labelpad=7)
ax.set_xlim(-0.3, 7.5)
ax.set_ylim(-2, 102)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(25))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fn = 'fig_mortality_curve.png'
plt.savefig(fn, dpi=300)
plt.savefig(os.path.join('plots', fn), dpi=300)
plt.close()
output_files.append(fn)

# ==============================================================================
# 2. STANDALONE FIGURE: Bee Survival Curve (Paper Style)
# ==============================================================================
fig, ax = plt.subplots(figsize=(6.8, 5.4), dpi=300)
surv_se_fit = np.std(residuals_surv)

ax.fill_between(t_dense, np.clip(fit_surv_curve - 1.96*surv_se_fit, 0, 100), np.clip(fit_surv_curve + 1.96*surv_se_fit, 0, 100),
                color='#ff9900', alpha=0.15, label='95% Confidence Band')
ax.plot(t_dense, fit_surv_curve, color='#e67300', linewidth=2.4, label='Survival Model Fit')
ax.errorbar(day_indices, survival_pct, yerr=survival_se, fmt='o', color='#e67300',
            ecolor='#e67300', elinewidth=1.6, capsize=4, capthick=1.6, markersize=7,
            markerfacecolor='#ff9900', markeredgecolor='#cc5200', label='Observed (Mean ± SE)')

ax.axhline(50, color='#333333', linestyle=':', linewidth=1.2)
eq_surv_text = f"$Y = {S0_surv:.1f} \\times e^{{-{a_surv:.4f} X^{{{b_surv:.2f}}}}}$\n$R^2 = {r2_surv:.4f}$"
ax.text(0.06, 0.32, eq_surv_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#e0e0e0', alpha=0.85))

ax.set_xlabel('Observation Time / Days', fontsize=13, fontweight='bold', labelpad=7)
ax.set_ylabel('Bee Survival / %', fontsize=13, fontweight='bold', labelpad=7)
ax.set_xlim(-0.3, 7.5)
ax.set_ylim(-2, 105)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(25))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fn = 'fig_survival_curve.png'
plt.savefig(fn, dpi=300)
plt.savefig(os.path.join('plots', fn), dpi=300)
plt.close()
output_files.append(fn)

# ==============================================================================
# 3. STANDALONE FIGURE: Kaplan-Meier Survival Analysis
# ==============================================================================
fig, ax = plt.subplots(figsize=(6.8, 5.4), dpi=300)

km_timeline = kmf.survival_function_.index
km_vals = kmf.survival_function_['KM_estimate'].values
ci_df = kmf.confidence_interval_survival_function_

ax.step(km_timeline, km_vals, where='post', color='#0066cc', linewidth=2.4, label='Kaplan-Meier Estimate')
ax.fill_between(km_timeline, ci_df.iloc[:, 0], ci_df.iloc[:, 1], step='post', color='#0066cc', alpha=0.15, label='95% Confidence Interval')

censored_days = durations[events == 0]
censored_surv = [kmf.survival_function_at_times(d).values[0] for d in censored_days]
ax.plot(censored_days, censored_surv, '+', color='#002244', markersize=9, markeredgewidth=2, label=f'Censored (n={len(censored_days)})')

ax.axhline(0.5, color='#333333', linestyle=':', linewidth=1.2)
ax.set_xlabel('Observation Time / Days', fontsize=13, fontweight='bold', labelpad=7)
ax.set_ylabel('Survival Probability', fontsize=13, fontweight='bold', labelpad=7)
ax.set_xlim(-0.2, 7.5)
ax.set_ylim(-0.02, 1.05)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='lower left', frameon=False, fontsize=10.5)

plt.tight_layout()
fn = 'fig_kaplan_meier.png'
plt.savefig(fn, dpi=300)
plt.savefig(os.path.join('plots', fn), dpi=300)
plt.close()
output_files.append(fn)

# ==============================================================================
# 4. STANDALONE FIGURE: Individual Bee Status Matrix Heatmap
# ==============================================================================
fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=300)

state_num_map = {'A': 1, 'B': 2, 'D': 3}
color_map = {1: '#2ca02c', 2: '#ff7f0e', 3: '#d62728'}

mat = np.zeros((num_bees, num_days))
for i, b_states in enumerate(states_matrix):
    for j, s in enumerate(b_states):
        mat[i, j] = state_num_map.get(s, 1)

for i in range(num_bees):
    for j in range(num_days):
        val = mat[i, j]
        c = color_map[val]
        ax.add_patch(plt.Rectangle((j, num_bees - 1 - i), 0.90, 0.85, color=c, ec='white', lw=1.5, zorder=2))
        ax.text(j + 0.45, num_bees - 1 - i + 0.425, states_matrix[i][j], ha='center', va='center',
                color='white', fontweight='bold', fontsize=11, zorder=3)

ax.set_xlim(-0.1, float(num_days))
ax.set_ylim(-0.1, num_bees)
ax.set_xticks(np.arange(num_days) + 0.45)
ax.set_xticklabels([f'Day {i}' for i in range(num_days)], fontsize=11.5, fontweight='bold')
ax.set_yticks(np.arange(num_bees) + 0.425)
ax.set_yticklabels([f'Bee {num_bees - i}' for i in range(num_bees)], fontsize=10.5, fontweight='bold')

ax.set_xlabel('Observation Day', fontsize=12.5, fontweight='bold', labelpad=7)
ax.set_ylabel('Individual Bee ID', fontsize=12.5, fontweight='bold', labelpad=7)

legend_elements = [
    Patch(facecolor='#2ca02c', edgecolor='none', label='Alive (A)'),
    Patch(facecolor='#ff7f0e', edgecolor='none', label='Behavioral Change (B)'),
    Patch(facecolor='#d62728', edgecolor='none', label='Dead (D)')
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.12),
          ncol=3, frameon=False, fontsize=10.5)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(left=False, bottom=False)

plt.tight_layout()
fn = 'fig_individual_bees_status.png'
plt.savefig(fn, dpi=300)
plt.savefig(os.path.join('plots', fn), dpi=300)
plt.close()
output_files.append(fn)

# ==============================================================================
# 5. STANDALONE FIGURE: Environmental Conditions (Temp & Humidity Dual Axis)
# ==============================================================================
fig, ax1 = plt.subplots(figsize=(7.0, 5.2), dpi=300)

x_days = np.arange(num_days) # Day 0 to Day 7
color_temp = '#d95f02'
ax1.set_xlabel('Observation Time / Days', fontsize=12.5, fontweight='bold', labelpad=7)
ax1.set_ylabel('Temperature / °C', color=color_temp, fontsize=12.5, fontweight='bold', labelpad=7)
l1 = ax1.plot(x_days, temps, color=color_temp, marker='o', linewidth=2.2, markersize=7.5, label='Temperature (°C)')
ax1.tick_params(axis='y', labelcolor=color_temp)
ax1.set_ylim(22.0, 26.0)

ax2 = ax1.twinx()
color_hum = '#1f78b4'
ax2.set_ylabel('Relative Humidity / %', color=color_hum, fontsize=12.5, fontweight='bold', labelpad=7)
l2 = ax2.plot(x_days, humidities, color=color_hum, marker='s', linewidth=2.2, markersize=7.5, linestyle='--', label='Relative Humidity (%)')
ax2.tick_params(axis='y', labelcolor=color_hum)
ax2.set_ylim(40, 75)

ax1.set_xlim(-0.3, 7.3)
ax1.xaxis.set_major_locator(MultipleLocator(1))
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

lines = l1 + l2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', frameon=True, framealpha=0.9, facecolor='white', edgecolor='none', fontsize=10.5)

plt.tight_layout()
fn = 'fig_environmental_conditions.png'
plt.savefig(fn, dpi=300)
plt.savefig(os.path.join('plots', fn), dpi=300)
plt.close()
output_files.append(fn)

# ==============================================================================
# 6. COMBINED MULTI-PANEL FIGURE (Panels A, B, C, D)
# ==============================================================================
fig = plt.figure(figsize=(14, 11), dpi=300)
gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.25)

# Panel A: Mortality Curve
ax_a = fig.add_subplot(gs[0, 0])
ax_a.fill_between(t_dense, np.clip(fit_mort_curve - 1.96*fit_se, 0, 100), np.clip(fit_mort_curve + 1.96*fit_se, 0, 100),
                  color='#e60000', alpha=0.12)
ax_a.plot(t_dense, fit_mort_curve, color='#e60000', linewidth=2.4)
ax_a.errorbar(day_indices, mortality_pct, yerr=mortality_se, fmt='o', color='#e60000',
              ecolor='#e60000', elinewidth=1.6, capsize=4, capthick=1.6, markersize=6.5,
              markerfacecolor='#e60000', markeredgecolor='#b30000')
ax_a.axhline(50, color='#333333', linestyle=':', linewidth=1.2)
ax_a.text(0.06, 0.90, f"$Y = \\frac{{{L_fit:.1f}}}{{1 + e^{{-{k_fit:.2f}(X - {t0_fit:.1f})}}}}$\n$R^2 = {r2_mort:.4f}$",
          transform=ax_a.transAxes, fontsize=10.5, verticalalignment='top')
ax_a.set_xlabel('Observation Time / Days', fontsize=11.5, fontweight='bold')
ax_a.set_ylabel('Mortality / %', fontsize=11.5, fontweight='bold')
ax_a.set_xlim(-0.3, 7.5)
ax_a.set_ylim(-2, 102)
ax_a.xaxis.set_major_locator(MultipleLocator(1))
ax_a.yaxis.set_major_locator(MultipleLocator(25))
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.text(-0.14, 1.04, 'A', transform=ax_a.transAxes, fontsize=15, fontweight='bold')

# Panel B: Survival Curve
ax_b = fig.add_subplot(gs[0, 1])
ax_b.fill_between(t_dense, np.clip(fit_surv_curve - 1.96*surv_se_fit, 0, 100), np.clip(fit_surv_curve + 1.96*surv_se_fit, 0, 100),
                  color='#ff9900', alpha=0.15)
ax_b.plot(t_dense, fit_surv_curve, color='#e67300', linewidth=2.4)
ax_b.errorbar(day_indices, survival_pct, yerr=survival_se, fmt='o', color='#e67300',
              ecolor='#e67300', elinewidth=1.6, capsize=4, capthick=1.6, markersize=6.5,
              markerfacecolor='#ff9900', markeredgecolor='#cc5200')
ax_b.axhline(50, color='#333333', linestyle=':', linewidth=1.2)
ax_b.text(0.06, 0.32, f"$Y = {S0_surv:.1f} \\times e^{{-{a_surv:.4f} X^{{{b_surv:.2f}}}}}$\n$R^2 = {r2_surv:.4f}$",
          transform=ax_b.transAxes, fontsize=10.5, verticalalignment='top')
ax_b.set_xlabel('Observation Time / Days', fontsize=11.5, fontweight='bold')
ax_b.set_ylabel('Bee Survival / %', fontsize=11.5, fontweight='bold')
ax_b.set_xlim(-0.3, 7.5)
ax_b.set_ylim(-2, 105)
ax_b.xaxis.set_major_locator(MultipleLocator(1))
ax_b.yaxis.set_major_locator(MultipleLocator(25))
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.text(-0.14, 1.04, 'B', transform=ax_b.transAxes, fontsize=15, fontweight='bold')

# Panel C: Kaplan-Meier Survival Step Function
ax_c = fig.add_subplot(gs[1, 0])
ax_c.step(km_timeline, km_vals, where='post', color='#0066cc', linewidth=2.4, label='KM Estimate')
ax_c.fill_between(km_timeline, ci_df.iloc[:, 0], ci_df.iloc[:, 1], step='post', color='#0066cc', alpha=0.15, label='95% CI')
ax_c.plot(censored_days, censored_surv, '+', color='#002244', markersize=8.5, markeredgewidth=2, label=f'Censored (n={len(censored_days)})')
ax_c.axhline(0.5, color='#333333', linestyle=':', linewidth=1.2)
ax_c.set_xlabel('Observation Time / Days', fontsize=11.5, fontweight='bold')
ax_c.set_ylabel('Survival Probability', fontsize=11.5, fontweight='bold')
ax_c.set_xlim(-0.2, 7.5)
ax_c.set_ylim(-0.02, 1.05)
ax_c.xaxis.set_major_locator(MultipleLocator(1))
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
ax_c.legend(loc='lower left', frameon=False, fontsize=9.5)
ax_c.text(-0.14, 1.04, 'C', transform=ax_c.transAxes, fontsize=15, fontweight='bold')

# Panel D: Environmental Monitoring
ax_d1 = fig.add_subplot(gs[1, 1])
ax_d1.set_xlabel('Observation Time / Days', fontsize=11.5, fontweight='bold')
ax_d1.set_ylabel('Temperature / °C', color=color_temp, fontsize=11.5, fontweight='bold')
l1 = ax_d1.plot(x_days, temps, color=color_temp, marker='o', linewidth=2.2, markersize=6.5, label='Temperature (°C)')
ax_d1.tick_params(axis='y', labelcolor=color_temp)
ax_d1.set_ylim(22.0, 26.0)

ax_d2 = ax_d1.twinx()
ax_d2.set_ylabel('Relative Humidity / %', color=color_hum, fontsize=11.5, fontweight='bold')
l2 = ax_d2.plot(x_days, humidities, color=color_hum, marker='s', linewidth=2.2, markersize=6.5, linestyle='--', label='Humidity (%)')
ax_d2.tick_params(axis='y', labelcolor=color_hum)
ax_d2.set_ylim(40, 75)

ax_d1.set_xlim(-0.3, 7.3)
ax_d1.xaxis.set_major_locator(MultipleLocator(1))
ax_d1.spines['top'].set_visible(False)
ax_d2.spines['top'].set_visible(False)
lines_comb = l1 + l2
ax_d1.legend(lines_comb, [l.get_label() for l in lines_comb], loc='upper left', frameon=True, framealpha=0.9, facecolor='white', edgecolor='none', fontsize=9.5)
ax_d1.text(-0.14, 1.04, 'D', transform=ax_d1.transAxes, fontsize=15, fontweight='bold')

fn = 'fig_combined_publication_summary.png'
plt.savefig(fn, dpi=300)
plt.savefig(os.path.join('plots', fn), dpi=300)
plt.close()
output_files.append(fn)

# Clean up any leftover old pdf files if needed
for old_pdf in [f for f in os.listdir('.') if f.endswith('.pdf') and f.startswith('fig_')]:
    try:
        os.remove(old_pdf)
    except:
        pass

for old_pdf in [f for f in os.listdir('plots') if f.endswith('.pdf') and f.startswith('fig_')]:
    try:
        os.remove(os.path.join('plots', old_pdf))
    except:
        pass

print("ALL_PLOTS_UPDATED_SUCCESSFULLY_NO_PDF")
for f in output_files:
    print(f"Generated PNG: {f} and plots/{f}")
