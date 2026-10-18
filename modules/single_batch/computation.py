"""
Single Batch Analytical Modeling & Mathematical Computations.
Calculates time progression, Resurrected Bee states correction,
non-linear curve fitting (Logistic and Weibull-type decay),
Kaplan-Meier estimates, and Finney Probit regression.
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm, linregress
from lifelines import KaplanMeierFitter

from modules.parser import parse_time_data


def compute_single_batch_data(df, settings=None):
    """
    Parses raw dataframe observations and calculates mathematical models,
    SE bands, residuals, Kaplan-Meier fitter, and Probit parameters.
    """
    if settings is None:
        settings = {}

    abbott_c = settings.get('abbott_c', 0.0)

    times_numeric, raw_times, time_col = parse_time_data(df)
    elapsed_days = times_numeric
    day_labels = raw_times
    num_days = len(df)
    max_time = elapsed_days[-1] if len(elapsed_days) > 0 else 1.0

    bee_cols = [c for c in df.columns if str(c).lower().startswith('bee')]
    num_bees = len(bee_cols)

    temps_raw = df['Temp (°C)'].tolist() if 'Temp (°C)' in df.columns else [0] * num_days
    humidities_raw = df['Humidity'].tolist() if 'Humidity' in df.columns else [0] * num_days
    temps = [float(x) if str(x).replace('.', '', 1).isdigit() else 0.0 for x in temps_raw]
    humidities = [float(x) if str(x).replace('.', '', 1).isdigit() else 0.0 for x in humidities_raw]

    cons_cols = [c for c in df.columns if str(c).startswith('Cons ') and '(µL)' in str(c)]
    cons_matrix = []
    for c in cons_cols:
        raw_vals = df[c].tolist()
        cons_matrix.append([float(x) if str(x).replace('.', '', 1).isdigit() else 0.0 for x in raw_vals])

    states_matrix = []
    for bc in bee_cols:
        states_matrix.append(df[bc].fillna("A").tolist())

    # Map each bee to its exact consumption column
    cons_cols = []
    cons_matrix = []
    for bc in bee_cols:
        cons_candidates = [
            bc.replace("Bee ", "Cons ") + " (µL)",
            bc.replace("Bee ", "Cons ") + " (uL)",
            bc.replace("Bee", "Cons"),
            bc.replace("Bee ", "Cons "),
            bc.replace("Bee_", "Cons_"),
            bc.replace("bee", "cons")
        ]
        c_found = False
        for cand in cons_candidates:
            matched = [c for c in df.columns if cand.lower() in str(c).lower()]
            if matched:
                raw_c = pd.to_numeric(df[matched[0]], errors='coerce').fillna(0.0).tolist()
                cons_matrix.append(raw_c)
                cons_cols.append(matched[0])
                c_found = True
                break
        if not c_found:
            cons_matrix.append([0.0] * num_days)
            cons_cols.append(f"Cons {bc} (µL)")

    has_consumption = any(any(v > 0 for v in b_c) for b_c in cons_matrix) if cons_matrix else False

    durations = []
    events = []

    def get_latest_state_single(bee_states, up_to_day_idx):
        for k in range(up_to_day_idx, -1, -1):
            state = str(bee_states[k]).strip().upper()
            if state in ['A', 'B', 'D']:
                return state
        return 'A'

    for b_states in states_matrix:
        final_state = get_latest_state_single(b_states, num_days - 1)
        if final_state == 'D':
            d_star = num_days - 1
            for k in range(num_days - 1, -1, -1):
                if get_latest_state_single(b_states, k) == 'D':
                    d_star = k
                else:
                    break
            dur = float(elapsed_days[d_star]) if d_star < len(elapsed_days) else float(elapsed_days[-1])
            durations.append(max(dur, 0.5))
            events.append(1)
        else:
            durations.append(float(elapsed_days[-1]) if num_days > 0 else 1.0)
            events.append(0)

    durations = np.array(durations)
    events = np.array(events)

    alive_counts = []
    dead_counts = []

    for d_idx in range(num_days):
        n_dead = sum(1 for states in states_matrix if get_latest_state_single(states, d_idx) == 'D')
        alive_counts.append(num_bees - n_dead)
        dead_counts.append(n_dead)

    alive_counts = np.array(alive_counts)
    dead_counts = np.array(dead_counts)
    survival_pct = alive_counts / num_bees * 100.0 if num_bees > 0 else np.zeros(num_days)
    mortality_pct = dead_counts / num_bees * 100.0 if num_bees > 0 else np.zeros(num_days)

    # Daily consumption calculations across all bees vs surviving bees
    daily_all_cons = []
    daily_mean_cons = []
    daily_sem_cons = []
    daily_alive_mean_cons = []
    daily_alive_sem_cons = []

    for d_idx in range(num_days):
        day_vals = [cons_matrix[b_idx][d_idx] for b_idx in range(len(cons_matrix))] if cons_matrix else []
        daily_all_cons.append(day_vals)
        if day_vals:
            m_val = float(np.mean(day_vals))
            sem_val = float(np.std(day_vals) / np.sqrt(len(day_vals))) if len(day_vals) > 1 else 0.0
        else:
            m_val, sem_val = 0.0, 0.0
        daily_mean_cons.append(m_val)
        daily_sem_cons.append(sem_val)

        alive_indices = [b_idx for b_idx in range(len(cons_matrix)) if get_latest_state_single(states_matrix[b_idx], d_idx) != 'D'] if cons_matrix else []
        if alive_indices:
            alive_day_vals = [cons_matrix[b_idx][d_idx] for b_idx in alive_indices]
            al_m = float(np.mean(alive_day_vals))
            al_sem = float(np.std(alive_day_vals) / np.sqrt(len(alive_day_vals))) if len(alive_day_vals) > 1 else 0.0
        else:
            al_m, al_sem = 0.0, 0.0
        daily_alive_mean_cons.append(al_m)
        daily_alive_sem_cons.append(al_sem)

    daily_mean_cons = np.array(daily_mean_cons)
    daily_sem_cons = np.array(daily_sem_cons)
    daily_alive_mean_cons = np.array(daily_alive_mean_cons)
    daily_alive_sem_cons = np.array(daily_alive_sem_cons)
    cumulative_mean_cons = np.cumsum(daily_mean_cons) if len(daily_mean_cons) > 0 else np.array([])
    cumulative_cons_matrix = np.cumsum(np.array(cons_matrix), axis=1).tolist() if len(cons_matrix) > 0 and num_days > 0 else []

    # Feeding cessation durations & events for Kaplan-Meier analysis
    feeding_durations = []
    feeding_events = []
    mean_cons_per_bee = []
    total_cons_per_bee = []

    for idx_b in range(num_bees):
        b_c = cons_matrix[idx_b] if idx_b < len(cons_matrix) else []
        final_st = get_latest_state_single(states_matrix[idx_b], num_days - 1)
        if b_c and any(v > 0 for v in b_c):
            pos_indices = [k for k, v in enumerate(b_c) if v > 0]
            last_feed_idx = max(pos_indices)
            if last_feed_idx < num_days - 1:
                cess_idx = min(last_feed_idx + 1, num_days - 1)
                dur_feed = float(elapsed_days[cess_idx]) if cess_idx < len(elapsed_days) else float(elapsed_days[-1])
                ev_feed = 1
            else:
                dur_feed = float(elapsed_days[-1]) if len(elapsed_days) > 0 else 1.0
                ev_feed = 0
            pos_vals = [v for v in b_c if v > 0]
            mean_cons_per_bee.append(float(np.mean(pos_vals)) if pos_vals else 0.0)
            total_cons_per_bee.append(float(np.sum(b_c)))
        else:
            dur_feed = float(elapsed_days[0]) if len(elapsed_days) > 0 else 1.0
            ev_feed = 1 if final_st == 'D' else 0
            mean_cons_per_bee.append(0.0)
            total_cons_per_bee.append(0.0)
        feeding_durations.append(max(dur_feed, 0.5))
        feeding_events.append(ev_feed)

    feeding_durations = np.array(feeding_durations)
    feeding_events = np.array(feeding_events)

    kmf_feeding = KaplanMeierFitter()
    if len(feeding_durations) > 0 and has_consumption:
        kmf_feeding.fit(feeding_durations, event_observed=feeding_events, timeline=np.linspace(0, max(elapsed_days) + 1, max(int(max(elapsed_days) * 20), 140)))

    # Correlation between Mean Daily Consumption and Survival Duration
    if (len(mean_cons_per_bee) > 2 and len(durations) == len(mean_cons_per_bee) and 
        np.std(mean_cons_per_bee) > 1e-5 and np.std(durations) > 1e-5):
        from scipy.stats import pearsonr
        try:
            r_val, p_val = pearsonr(mean_cons_per_bee, durations)
            res_corr = linregress(mean_cons_per_bee, durations)
            cons_corr_slope = res_corr.slope
            cons_corr_intercept = res_corr.intercept
            cons_corr_r = r_val if np.isfinite(r_val) else 0.0
            cons_corr_p = p_val if np.isfinite(p_val) else 1.0
        except Exception:
            cons_corr_slope, cons_corr_intercept, cons_corr_r, cons_corr_p = 0.0, 0.0, 0.0, 1.0
    else:
        cons_corr_slope, cons_corr_intercept, cons_corr_r, cons_corr_p = 0.0, 0.0, 0.0, 1.0

    # Abbott Corrected Mortality
    abbott_corrected_pct = np.zeros_like(mortality_pct)
    if abbott_c < 100:
        abbott_corrected_pct = (mortality_pct - abbott_c) / (100.0 - abbott_c) * 100.0
        abbott_corrected_pct = np.clip(abbott_corrected_pct, 0, 100)

    # Finney empirical probit transformation
    p_lower = 0.5 / max(num_bees, 10)
    p_upper = 1.0 - p_lower
    probit_vals = np.array([norm.ppf(np.clip(m / 100.0, p_lower, p_upper)) + 5.0 for m in abbott_corrected_pct])

    first_nonzero_idx = next((i for i, m in enumerate(abbott_corrected_pct) if m > 0), len(abbott_corrected_pct) - 1)
    start_idx = max(0, first_nonzero_idx - 1)

    reg_log_time = []
    reg_probit_y = []
    for i in range(start_idx, len(elapsed_days)):
        d = elapsed_days[i]
        if d > 0:
            reg_log_time.append(np.log10(d))
            reg_probit_y.append(probit_vals[i])

    log_time = np.array([np.log10(d) for d in elapsed_days if d > 0])
    probit_y = probit_vals[:len(log_time)]

    if len(reg_log_time) > 1:
        res = linregress(reg_log_time, reg_probit_y)
        probit_a, probit_b, probit_r = res.intercept, res.slope, res.rvalue
    else:
        probit_a, probit_b, probit_r = 0, 0, 0

    mortality_se = np.sqrt((mortality_pct / 100.0) * (1.0 - mortality_pct / 100.0) / max(1, num_bees)) * 100.0
    survival_se = np.sqrt((survival_pct / 100.0) * (1.0 - survival_pct / 100.0) / max(1, num_bees)) * 100.0

    def logistic_mort(t, L, k, t0):
        return L / (1.0 + np.exp(-k * (t - t0)))

    try:
        popt_mort, _ = curve_fit(
            logistic_mort, elapsed_days, mortality_pct,
            p0=[60.0, 0.5, 5.5],
            bounds=([20.0, 0.01, 0.0], [100.0, 5.0, 15.0]),
            maxfev=10000
        )
    except Exception:
        popt_mort = [60.0, 0.5, 5.5]

    L_fit, k_fit, t0_fit = popt_mort
    residuals_mort = mortality_pct - logistic_mort(elapsed_days, *popt_mort)
    ss_res_mort = np.sum(residuals_mort**2)
    ss_tot_mort = np.sum((mortality_pct - np.mean(mortality_pct))**2)
    r2_mort = 1.0 - (ss_res_mort / ss_tot_mort) if ss_tot_mort != 0 else 1.0

    def survival_decay(t, S0, a, b):
        return S0 * np.exp(-a * (t**b))

    try:
        popt_surv, _ = curve_fit(
            survival_decay, elapsed_days, survival_pct,
            p0=[78.57, 0.001, 3.0],
            bounds=([50.0, 1e-6, 0.1], [100.0, 2.0, 10.0]),
            maxfev=10000
        )
    except Exception:
        popt_surv = [78.57, 0.001, 3.0]

    S0_surv, a_surv, b_surv = popt_surv
    residuals_surv = survival_pct - survival_decay(elapsed_days, *popt_surv)
    ss_res_surv = np.sum(residuals_surv**2)
    ss_tot_surv = np.sum((survival_pct - np.mean(survival_pct))**2)
    r2_surv = 1.0 - (ss_res_surv / ss_tot_surv) if ss_tot_surv != 0 else 1.0

    kmf = KaplanMeierFitter()
    if len(durations) > 0:
        kmf.fit(durations, event_observed=events, timeline=np.linspace(0, max(elapsed_days) + 1, max(int(max(elapsed_days) * 20), 140)))

    t_dense = np.linspace(0, max_time * 1.05, 250) if num_days > 0 else np.array([])
    fit_mort_curve = logistic_mort(t_dense, *popt_mort) if num_days > 0 else np.array([])
    fit_surv_curve = survival_decay(t_dense, *popt_surv) if num_days > 0 else np.array([])

    return {
        'num_days': num_days,
        'num_bees': num_bees,
        'elapsed_days': elapsed_days,
        'day_labels': day_labels,
        'max_time': max_time,
        'bee_cols': bee_cols,
        'cons_cols': cons_cols,
        'cons_matrix': cons_matrix,
        'states_matrix': states_matrix,
        'temps': temps,
        'humidities': humidities,
        'durations': durations,
        'events': events,
        'alive_counts': alive_counts,
        'dead_counts': dead_counts,
        'survival_pct': survival_pct,
        'mortality_pct': mortality_pct,
        'abbott_corrected_pct': abbott_corrected_pct,
        'mortality_se': mortality_se,
        'survival_se': survival_se,
        't_dense': t_dense,
        'fit_mort_curve': fit_mort_curve,
        'fit_surv_curve': fit_surv_curve,
        'residuals_mort': residuals_mort,
        'residuals_surv': residuals_surv,
        'L_fit': L_fit,
        'k_fit': k_fit,
        't0_fit': t0_fit,
        'r2_mort': r2_mort,
        'S0_surv': S0_surv,
        'a_surv': a_surv,
        'b_surv': b_surv,
        'r2_surv': r2_surv,
        'kmf': kmf,
        'probit_a': probit_a,
        'probit_b': probit_b,
        'probit_r': probit_r,
        'log_time': log_time,
        'probit_y': probit_y,
        'daily_all_cons': daily_all_cons,
        'daily_mean_cons': daily_mean_cons,
        'daily_sem_cons': daily_sem_cons,
        'daily_alive_mean_cons': daily_alive_mean_cons,
        'daily_alive_sem_cons': daily_alive_sem_cons,
        'cumulative_mean_cons': cumulative_mean_cons,
        'cumulative_cons_matrix': cumulative_cons_matrix,
        'has_consumption': has_consumption,
        'feeding_durations': feeding_durations,
        'feeding_events': feeding_events,
        'kmf_feeding': kmf_feeding,
        'mean_cons_per_bee': mean_cons_per_bee,
        'total_cons_per_bee': total_cons_per_bee,
        'cons_corr_slope': cons_corr_slope,
        'cons_corr_intercept': cons_corr_intercept,
        'cons_corr_r': cons_corr_r,
        'cons_corr_p': cons_corr_p
    }
