"""
Core Batch DataFrame Parser.
Processes single and multi-batch CSV tables, applying Abbott's formula,
individual bee tracking, and Probit linear regression.
Enforces Post-2h Starvation Rule (Excluding d0 / Time Zero).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, linregress

from modules.core.time_parser import parse_time_data, filter_post_starvation_data, is_time_zero
from modules.core.probit_engine import calculate_lt_percentiles


def parse_batch_df(raw_df, abbott_c=0.0, batch_name=None):
    """
    Parses a single batch DataFrame into mortality, survival, Probit regression,
    and individual bee time-to-event records.
    Strictly applies the Resurrected Bee observation correction rule and
    the post-2h starvation starting rule (excluding time zero / d0).
    """
    if isinstance(abbott_c, str):
        try:
            abbott_c = float(abbott_c)
        except ValueError:
            if batch_name is None:
                batch_name = abbott_c
            abbott_c = 0.0
    else:
        try:
            abbott_c = float(abbott_c)
        except (ValueError, TypeError):
            abbott_c = 0.0

    df_copy = raw_df.copy()
    df_copy.columns = [str(c).strip() for c in df_copy.columns]

    # Handle wide format where bee_id is a column (e.g. 2026-30_export.csv)
    if 'bee_id' in df_copy.columns:
        # Exclude pre-starvation time zero columns (d0, Day 0)
        t0_cols = [c for c in df_copy.columns if is_time_zero(c)]
        if t0_cols:
            df_copy.drop(columns=t0_cols, inplace=True)
        df_copy = df_copy.set_index('bee_id').T
        df_copy.reset_index(inplace=True)
        df_copy.rename(columns={'index': 'Day'}, inplace=True)
        df_copy['Day'] = df_copy['Day'].apply(lambda x: f"Day {str(x)[1:]}" if str(x).lower().startswith('d') and str(x)[1:].isdigit() else str(x))

        cols = list(df_copy.columns)
        for i in range(1, len(cols)):
            col_name = str(cols[i])
            if col_name.isdigit():
                cols[i] = f"Bee {col_name}"
            elif col_name == 'Temp':
                cols[i] = 'Temp (°C)'
        df_copy.columns = cols

    # Filter out time zero rows
    df_copy = filter_post_starvation_data(df_copy)

    batch_times, day_labels, time_col = parse_time_data(df_copy)
    num_days = len(df_copy)

    if len(batch_times) > 0:
        start_time = batch_times[0]
        batch_elapsed_days = np.array([t - start_time + 1.0 for t in batch_times])
    else:
        batch_elapsed_days = np.arange(1.0, num_days + 1.0)
        start_time = 1.0

    mort_col = next((c for c in df_copy.columns if 'mortality' in c.lower() or 'dead' in c.lower()), None)

    if mort_col and not any(str(c).lower().startswith('bee') for c in df_copy.columns):
        raw_vals = pd.to_numeric(df_copy[mort_col], errors='coerce').fillna(0).values
        if len(raw_vals) > 0 and max(raw_vals) <= 1.0:
            mortality_pct = raw_vals * 100.0
            num_bees = 10
        elif len(raw_vals) > 0 and max(raw_vals) <= 100.0:
            mortality_pct = raw_vals
            num_bees = 10
        else:
            mortality_pct = np.zeros(num_days)
            num_bees = 10
        times_numeric = batch_elapsed_days
        bee_cols = []
    else:
        bee_cols = [c for c in df_copy.columns if str(c).lower().startswith('bee')]
        num_bees = len(bee_cols) if len(bee_cols) > 0 else 10
        times_numeric = batch_elapsed_days

        states_matrix = []
        for bc in bee_cols:
            states_matrix.append(df_copy[bc].fillna("A").tolist())

        def get_latest_state(bee_states, up_to_day_idx):
            for k in range(up_to_day_idx, -1, -1):
                state = str(bee_states[k]).strip().upper()
                if state in ['A', 'B', 'D']:
                    return state
            return 'A'

        dead_counts = []
        for d_idx in range(num_days):
            n_dead = sum(1 for states in states_matrix if get_latest_state(states, d_idx) == 'D')
            dead_counts.append(n_dead)

        dead_counts = np.array(dead_counts)
        mortality_pct = dead_counts / num_bees * 100.0 if num_bees > 0 else np.zeros(num_days)

    survival_pct = 100.0 - mortality_pct

    # Abbott Corrected Mortality
    abbott_corrected_pct = np.zeros_like(mortality_pct)
    if abbott_c < 100:
        abbott_corrected_pct = (mortality_pct - abbott_c) / (100.0 - abbott_c) * 100.0
        abbott_corrected_pct = np.clip(abbott_corrected_pct, 0, 100)

    # Finney Empirical Probit Transformation
    p_lower = 0.5 / max(num_bees, 10)
    p_upper = 1.0 - p_lower
    probit_vals = np.array([norm.ppf(np.clip(m / 100.0, p_lower, p_upper)) + 5.0 for m in abbott_corrected_pct])

    first_nonzero_idx = next((i for i, m in enumerate(abbott_corrected_pct) if m > 0), len(abbott_corrected_pct) - 1)
    start_idx = max(0, first_nonzero_idx - 1)

    reg_log_time = []
    reg_probit_y = []
    for i in range(start_idx, len(batch_elapsed_days)):
        d = batch_elapsed_days[i]
        if d > 0:
            reg_log_time.append(np.log10(d))
            reg_probit_y.append(probit_vals[i])

    plot_log_time = np.array([np.log10(d) for d in batch_elapsed_days if d > 0])
    plot_probit_y = probit_vals[:len(plot_log_time)]

    if len(reg_log_time) > 1:
        res = linregress(reg_log_time, reg_probit_y)
        a, b, r_val = res.intercept, res.slope, res.rvalue
        r_squared = r_val ** 2
    else:
        a, b, r_val, r_squared = 0.0, 0.0, 0.0, 0.0

    # Calculate default LT50 and LT90
    if b > 0:
        probit_50 = 5.0
        probit_90 = norm.ppf(0.90) + 5.0
        lt50_theoretical = 10.0 ** ((probit_50 - a) / b)
        lt90_theoretical = 10.0 ** ((probit_90 - a) / b)
        lt50_elapsed = lt50_theoretical - start_time if lt50_theoretical > start_time else 1.0
        lt90_elapsed = lt90_theoretical - start_time if lt90_theoretical > start_time else 1.0
    else:
        lt50_elapsed = np.nan
        lt90_elapsed = np.nan

    individual_bees = []
    if len(bee_cols) > 0:
        for idx_b, bc in enumerate(bee_cols):
            b_states = states_matrix[idx_b]
            final_state = get_latest_state(b_states, num_days - 1)

            cons_candidates = [
                bc.replace("Bee ", "Cons ") + " (µL)",
                bc.replace("Bee ", "Cons ") + " (uL)",
                bc.replace("Bee", "Cons")
            ]
            cons_vals = []
            for candidate in cons_candidates:
                matched = [c for c in df_copy.columns if candidate.lower() in str(c).lower()]
                if matched:
                    cons_vals = pd.to_numeric(df_copy[matched[0]], errors='coerce').fillna(0.0).tolist()
                    break

            if final_state == 'D':
                d_star = num_days - 1
                for k in range(num_days - 1, -1, -1):
                    if get_latest_state(b_states, k) == 'D':
                        d_star = k
                    else:
                        break
                dur = float(batch_elapsed_days[d_star]) if d_star < len(batch_elapsed_days) else float(batch_elapsed_days[-1])
                ev = 1
                alive_cons = cons_vals[:d_star + 1] if cons_vals else []
            else:
                dur = float(batch_elapsed_days[-1]) if len(batch_elapsed_days) > 0 else 1.0
                ev = 0
                alive_cons = cons_vals

            mean_cons = float(np.mean(alive_cons)) if len(alive_cons) > 0 else 0.0
            total_cons = float(np.sum(alive_cons)) if len(alive_cons) > 0 else 0.0

            temp_val = np.nan
            humid_val = np.nan
            for c in df_copy.columns:
                cl = str(c).lower()
                if 'temp' in cl:
                    try:
                        temp_val = float(pd.to_numeric(df_copy[c], errors='coerce').mean())
                    except Exception:
                        pass
                elif 'humid' in cl:
                    try:
                        humid_val = float(pd.to_numeric(df_copy[c], errors='coerce').mean())
                    except Exception:
                        pass

            treatment_val = "Batch"
            if 'Treatment' in df_copy.columns and pd.notna(df_copy['Treatment'].iloc[0]):
                treatment_val = str(df_copy['Treatment'].iloc[0]).strip()

            # Feeding cessation endpoint: find when bee permanently stops eating (Cons <= 0)
            if cons_vals and any(v > 0 for v in cons_vals):
                pos_indices = [k for k, v in enumerate(cons_vals) if v > 0]
                last_feed_idx = max(pos_indices)
                if last_feed_idx < num_days - 1:
                    cess_idx = min(last_feed_idx + 1, num_days - 1)
                    dur_feed = float(batch_elapsed_days[cess_idx])
                    ev_feed = 1
                else:
                    dur_feed = float(batch_elapsed_days[-1])
                    ev_feed = 0
            else:
                dur_feed = float(batch_elapsed_days[0]) if len(batch_elapsed_days) > 0 else 1.0
                ev_feed = 1 if final_state == 'D' else 0

            individual_bees.append({
                'bee_id': bc,
                'duration': max(dur, 0.5),
                'event': ev,
                'mean_consumption': mean_cons,
                'total_consumption': total_cons,
                'feeding_cessation_duration': max(dur_feed, 0.5),
                'feeding_cessation_event': ev_feed,
                'temp': temp_val if not np.isnan(temp_val) else 25.0,
                'humidity': humid_val if not np.isnan(humid_val) else 60.0,
                'treatment': treatment_val
            })

    # Extract complete batch consumption matrix and daily progression
    cons_matrix = []
    cons_cols = []
    if len(bee_cols) > 0:
        for idx_b, bc in enumerate(bee_cols):
            cons_candidates = [
                bc.replace("Bee ", "Cons ") + " (µL)",
                bc.replace("Bee ", "Cons ") + " (uL)",
                bc.replace("Bee", "Cons"),
                bc.replace("Bee ", "Cons "),
                bc.replace("Bee_", "Cons_"),
                bc.replace("bee", "cons")
            ]
            c_found = False
            for candidate in cons_candidates:
                matched = [c for c in df_copy.columns if candidate.lower() in str(c).lower()]
                if matched:
                    raw_c = pd.to_numeric(df_copy[matched[0]], errors='coerce').fillna(0.0).tolist()
                    cons_matrix.append(raw_c)
                    cons_cols.append(matched[0])
                    c_found = True
                    break
            if not c_found:
                cons_matrix.append([0.0] * num_days)
                cons_cols.append(f"Cons {bc} (µL)")

    has_consumption = any(any(v > 0 for v in b_c) for b_c in cons_matrix) if cons_matrix else False

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

        alive_indices = [b_idx for b_idx in range(len(cons_matrix)) if get_latest_state(states_matrix[b_idx], d_idx) != 'D'] if cons_matrix else []
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

    # Calculate full default LT spectrum
    default_lt_records = calculate_lt_percentiles(
        intercept_a=a,
        slope_b=b,
        start_time=start_time,
        times_numeric=batch_elapsed_days,
        mortality_pct=abbott_corrected_pct,
        percentiles=(10, 20, 30, 40, 50, 60, 70, 80, 90)
    )
    lt_map = {r['label']: r['effective_elapsed_int'] for r in default_lt_records}

    return {
        'num_days': num_days,
        'num_bees': num_bees,
        'day_labels': day_labels,
        'times_numeric': times_numeric,
        'batch_elapsed_days': batch_elapsed_days,
        'start_time': start_time,
        'mortality_pct': mortality_pct,
        'abbott_corrected_pct': abbott_corrected_pct,
        'survival_pct': survival_pct,
        'probit_vals': probit_vals,
        'log_time': plot_log_time,
        'probit_y': plot_probit_y,
        'intercept_a': a,
        'slope_b': b,
        'r_squared': r_squared,
        'r_val': r_val,
        'lt10': lt_map.get('LT10', np.nan),
        'lt20': lt_map.get('LT20', np.nan),
        'lt30': lt_map.get('LT30', np.nan),
        'lt50': lt50_elapsed,
        'lt90': lt90_elapsed,
        'lt_records': default_lt_records,
        'lt_dict': lt_map,
        'individual_bees': pd.DataFrame(individual_bees),
        'cons_matrix': cons_matrix,
        'cons_cols': cons_cols,
        'daily_all_cons': daily_all_cons,
        'daily_mean_cons': daily_mean_cons,
        'daily_sem_cons': daily_sem_cons,
        'daily_alive_mean_cons': daily_alive_mean_cons,
        'daily_alive_sem_cons': daily_alive_sem_cons,
        'cumulative_mean_cons': cumulative_mean_cons,
        'has_consumption': has_consumption
    }


def standardize_observation_df(raw_df, default_treatment="sucrose solution 2 microliter"):
    """
    Standardizes any uploaded DataFrame into canonical observation sheet format:
    - Filters out pre-starvation time zero (d0, Day 0).
    - If wide format with 'bee_id', transposes to Day rows and Bee columns.
    - Ensures 'Temp (°C)', 'Treatment', 'Observations/notes', and 'Cons' columns exist.
    """
    if raw_df is None or raw_df.empty:
        return raw_df

    df_copy = raw_df.copy()
    df_copy.columns = [str(c).strip() for c in df_copy.columns]

    if 'bee_id' in df_copy.columns:
        t0_cols = [c for c in df_copy.columns if is_time_zero(c)]
        if t0_cols:
            df_copy.drop(columns=t0_cols, inplace=True)
        df_copy = df_copy.set_index('bee_id').T
        df_copy.reset_index(inplace=True)
        df_copy.rename(columns={'index': 'Day'}, inplace=True)
        df_copy['Day'] = df_copy['Day'].apply(lambda x: f"Day {str(x)[1:]}" if str(x).lower().startswith('d') and str(x)[1:].isdigit() else str(x))

        cols = list(df_copy.columns)
        for i in range(1, len(cols)):
            col_name = str(cols[i])
            if col_name.isdigit():
                cols[i] = f"Bee {col_name}"
            elif col_name == 'Temp':
                cols[i] = 'Temp (°C)'
        df_copy.columns = cols

        if 'Date/time' not in df_copy.columns:
            df_copy.insert(1, 'Date/time', '')

        bee_cols = [c for c in df_copy.columns if str(c).lower().startswith('bee')]
        for b in bee_cols:
            cons_col = b.replace("Bee ", "Cons ") + " (µL)"
            if cons_col not in df_copy.columns:
                df_copy.insert(df_copy.columns.get_loc(b) + 1, cons_col, "0.0")

        if 'Treatment' not in df_copy.columns:
            df_copy['Treatment'] = default_treatment or 'sucrose solution 2 microliter'
        if 'Observations/notes' not in df_copy.columns:
            df_copy['Observations/notes'] = ''

    df_copy = filter_post_starvation_data(df_copy)
    return df_copy
