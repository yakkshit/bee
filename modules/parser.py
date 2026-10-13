import re
import numpy as np
import pandas as pd
from scipy.stats import norm, linregress

DEFAULT_CSV = """Day,Date/time,Bee 1,Bee 2,Bee 3,Bee 4,Bee 5,Bee 6,Bee 7,Bee 8,Bee 9,Bee 10,Bee 11,Bee 12,Bee 13,Bee 14,Temp (°C),Humidity,Treatment,Observations/notes
Day 1,02.09.2026-16:00,A,D,A,A,A,A,A,A,D,A,A,A,A,D,23.3,55,Sucrose solution,
Day 2,03.09.2026-16:00,A,D,A,A,A,A,A,A,D,A,A,A,A,D,24.4,45,Sucrose solution,
Day 3,04.09.2026-16:00,A,D,A,A,B,A,A,A,D,A,A,A,A,D,24.7,58,Sucrose solution,
Day 4,05.09.2026-16:00,A,D,A,A,B,A,A,A,D,A,A,A,A,D,23.8,61,Sucrose solution,
Day 5,06.09.2026-16:00,A,D,D,A,B,A,A,A,D,A,A,D,A,D,23.7,51,Sucrose solution,
Day 6,07.09.2026-16:00,D,D,D,A,B,A,A,A,D,A,A,D,A,D,24.5,64,Sucrose solution,
Day 7,08.09.2026-16:00,D,D,D,B,B,A,A,A,D,A,A,D,A,D,24.8,62,Sucrose solution,
Day 8,09.09.2026-16:00,D,D,D,B,B,A,A,A,D,A,A,D,A,D,23.4,59,Sucrose solution,"""


def parse_time_data(df):
    """Parses the 'Day' column into continuous elapsed days starting at Day 1 = 1.0."""
    time_col_name = 'Day' if 'Day' in df.columns else ('Date/time' if 'Date/time' in df.columns else None)

    if time_col_name:
        raw_times = df[time_col_name].astype(str).tolist()
    else:
        raw_times = [f"Day {i+1}" for i in range(len(df))]

    def parse_to_days(t_str):
        t_str = str(t_str).strip()
        match = re.match(r"(?:Day|Dy|D)?\s*(\d+)(?:_(\d+)(?::(\d+))?)?", t_str, re.IGNORECASE)
        if match and match.group(1):
            d = int(match.group(1))
            # Treat single-digit observation indices without minutes (e.g. Day 1_2) as h = 0
            # Retain fractional hours only for explicit hours/minutes (e.g. Day 1_2:10 or Day 1_23)
            if match.group(3):
                h = int(match.group(2))
                m = int(match.group(3))
            elif match.group(2) and len(match.group(2)) > 1:
                h = int(match.group(2))
                m = 0
            else:
                h = 0
                m = 0
            return d + (h / 24.0) + (m / 1440.0)
        try:
            return float(t_str)
        except ValueError:
            return None

    numeric_times = []
    current_day = 1.0
    for t in raw_times:
        parsed = parse_to_days(t)
        if parsed is not None:
            current_day = parsed
            numeric_times.append(current_day)
        else:
            numeric_times.append(current_day)
            current_day += 1.0

    return np.array(numeric_times), raw_times, time_col_name


def parse_batch_df(raw_df, abbott_c=0.0):
    """
    Parses a single batch DataFrame into mortality, survival, Probit regression,
    and individual bee time-to-event records.
    Strictly applies the Resurrected Bee observation correction rule.
    """
    df_copy = raw_df.copy()
    df_copy.columns = [c.strip() for c in df_copy.columns]

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

        if len(bee_cols) > 0:
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
            mortality_pct = (dead_counts / num_bees) * 100.0 if num_bees > 0 else np.zeros(num_days)
        else:
            mortality_pct = np.zeros(num_days)

    abbott_corrected_pct = np.copy(mortality_pct)
    if abbott_c < 100:
        abbott_corrected_pct = (mortality_pct - abbott_c) / (100.0 - abbott_c) * 100.0
        abbott_corrected_pct = np.clip(abbott_corrected_pct, 0, 100)

    survival_pct = 100.0 - mortality_pct

    # Empirical discrete observation day where threshold is reached
    empirical_lt50 = np.nan
    empirical_lt90 = np.nan
    for idx in range(len(batch_elapsed_days)):
        if abbott_corrected_pct[idx] >= 50.0 and np.isnan(empirical_lt50):
            empirical_lt50 = batch_elapsed_days[idx]
        if abbott_corrected_pct[idx] >= 90.0 and np.isnan(empirical_lt90):
            empirical_lt90 = batch_elapsed_days[idx]

    # Standard Finney empirical probit transformation
    p_lower = 0.5 / max(num_bees, 10)
    p_upper = 1.0 - p_lower

    probit_vals = []
    for m in abbott_corrected_pct:
        p_val = m / 100.0
        p_clamped = np.clip(p_val, p_lower, p_upper)
        probit_vals.append(norm.ppf(p_clamped) + 5.0)
    probit_vals = np.array(probit_vals)

    first_nonzero_idx = next((i for i, m in enumerate(abbott_corrected_pct) if m > 0), len(abbott_corrected_pct) - 1)
    start_idx = max(0, first_nonzero_idx - 1)

    reg_log_time = []
    reg_probit_y = []
    for i in range(start_idx, len(batch_elapsed_days)):
        t = batch_elapsed_days[i]
        if t > 0:
            reg_log_time.append(np.log10(t))
            reg_probit_y.append(probit_vals[i])

    reg_log_time = np.array(reg_log_time)
    reg_probit_y = np.array(reg_probit_y)

    plot_log_time = []
    plot_probit_y = []
    for i, t in enumerate(batch_elapsed_days):
        if t > 0:
            plot_log_time.append(np.log10(t))
            plot_probit_y.append(probit_vals[i])
    plot_log_time = np.array(plot_log_time)
    plot_probit_y = np.array(plot_probit_y)

    if len(reg_log_time) > 1:
        res = linregress(reg_log_time, reg_probit_y)
        a, b, r_val = res.intercept, res.slope, res.rvalue
        r_squared = r_val ** 2
        if b > 0:
            probit_lt50 = 10 ** ((5.0 - a) / b)
            probit_lt90 = 10 ** ((6.28155 - a) / b)
        else:
            probit_lt50, probit_lt90 = np.nan, np.nan
    else:
        a, b, r_val, r_squared = 0.0, 0.0, 0.0, 0.0
        probit_lt50, probit_lt90 = np.nan, np.nan

    if not np.isnan(empirical_lt50):
        lt50 = empirical_lt50
    else:
        lt50 = probit_lt50

    if not np.isnan(empirical_lt90):
        lt90 = empirical_lt90
    else:
        lt90 = probit_lt90

    lt50_elapsed = int(np.ceil(lt50 - start_time)) if not np.isnan(lt50) else np.nan
    lt90_elapsed = int(np.ceil(lt90 - start_time)) if not np.isnan(lt90) else np.nan

    # Extract individual bee level survival data for Log-Rank and Cox Regression
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
                alive_cons = cons_vals[:d_star+1] if cons_vals else []
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

            individual_bees.append({
                'bee_id': bc,
                'duration': max(dur, 0.5),
                'event': ev,
                'mean_consumption': mean_cons,
                'total_consumption': total_cons,
                'temp': temp_val if not np.isnan(temp_val) else 25.0,
                'humidity': humid_val if not np.isnan(humid_val) else 60.0,
                'treatment': treatment_val
            })

    # Calculate full default LT spectrum (10, 20, 30, 40, 50, 60, 70, 80, 90)
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
        'lt50': lt50_elapsed,
        'lt90': lt90_elapsed,
        'lt_records': default_lt_records,
        'lt_dict': lt_map,
        'individual_bees': pd.DataFrame(individual_bees)
    }


def calculate_lt_percentiles(intercept_a, slope_b, start_time=1.0, times_numeric=None, mortality_pct=None, percentiles=(10, 20, 30, 40, 50, 60, 70, 80, 90)):
    """
    Computes Finney Probit-derived theoretical LT_X and observed empirical LT_X
    for arbitrary percentiles X in (0, 100).
    
    Formula:
        Probit(P) = norm.ppf(X / 100) + 5.0
        Theoretical LT_X = 10 ** ((Probit(P) - a) / b) if b > 0
        Theoretical Elapsed = Theoretical LT_X - start_time
        
        Empirical LT_X:
            Continuous or discrete observation day where mortality reaches or crosses X%.
    """
    results = []
    clean_p = sorted(list(set([float(p) for p in percentiles if 0 < float(p) < 100])))

    for p in clean_p:
        lbl = f"LT{int(p) if p.is_integer() else p}"
        p_ratio = np.clip(p / 100.0, 1e-4, 1.0 - 1e-4)
        probit_score = float(norm.ppf(p_ratio) + 5.0)

        # Theoretical Probit OLS estimate
        if slope_b > 0:
            theo_lt = float(10.0 ** ((probit_score - intercept_a) / slope_b))
            theo_elapsed = max(0.0, theo_lt - start_time)
            theo_elapsed_int = int(np.ceil(theo_elapsed))
        else:
            theo_lt = np.nan
            theo_elapsed = np.nan
            theo_elapsed_int = np.nan

        # Empirical bioassay observed estimate
        emp_lt = np.nan
        emp_elapsed = np.nan
        emp_elapsed_int = np.nan
        reached = False

        if times_numeric is not None and mortality_pct is not None and len(times_numeric) > 0 and len(mortality_pct) > 0:
            t_arr = np.array(times_numeric, dtype=float)
            m_arr = np.array(mortality_pct, dtype=float)

            if m_arr[0] >= p:
                emp_lt = float(t_arr[0])
                emp_elapsed = max(0.0, emp_lt - start_time)
                emp_elapsed_int = int(np.ceil(emp_elapsed))
                reached = True
            else:
                for idx in range(1, len(m_arr)):
                    if m_arr[idx] >= p:
                        t_prev = float(t_arr[idx - 1])
                        t_curr = float(t_arr[idx])
                        m_prev = float(m_arr[idx - 1])
                        m_curr = float(m_arr[idx])

                        if m_curr > m_prev:
                            emp_lt = t_prev + ((p - m_prev) / (m_curr - m_prev)) * (t_curr - t_prev)
                        else:
                            emp_lt = t_curr

                        emp_elapsed = max(0.0, emp_lt - start_time)
                        emp_elapsed_int = int(np.ceil(emp_elapsed))
                        reached = True
                        break

        # Effective elapsed days: prefer empirical observation if reached, fallback to theoretical model
        if not np.isnan(emp_elapsed_int):
            effective_elapsed_int = emp_elapsed_int
            status_desc = f"Observed ({emp_elapsed:.1f}d)"
        elif not np.isnan(theo_elapsed_int):
            effective_elapsed_int = theo_elapsed_int
            status_desc = f"Probit Model ({theo_elapsed:.1f}d)"
        else:
            effective_elapsed_int = np.nan
            status_desc = "Not Reached"

        results.append({
            'percentile': p,
            'label': lbl,
            'probit_score': probit_score,
            'theoretical_lt': theo_lt,
            'theoretical_elapsed': theo_elapsed,
            'theoretical_elapsed_int': theo_elapsed_int,
            'empirical_lt': emp_lt,
            'empirical_elapsed': emp_elapsed,
            'empirical_elapsed_int': emp_elapsed_int,
            'effective_elapsed_int': effective_elapsed_int,
            'status': status_desc,
            'reached': reached
        })

    return results


def extract_individual_bee_survival(df_in, batch_name="Batch", elapsed_days=None):
    """
    Extracts individual bee level survival data:
    - duration: Elapsed observation time to death or censoring
    - event: 1 (dead) or 0 (censored/alive)
    - mean_consumption: Average daily consumption (µL) while alive
    - total_consumption: Total consumption (µL) while alive
    - temp: Mean temperature (°C)
    - humidity: Mean humidity (%)
    Strictly adheres to the Resurrected Bee observation correction rule.
    """
    df_copy = df_in.copy()
    bee_cols = [c for c in df_copy.columns if str(c).lower().startswith('bee')]
    num_days = len(df_copy)

    if elapsed_days is None:
        batch_times, _, _ = parse_time_data(df_copy)
        if len(batch_times) > 0:
            start_t = batch_times[0]
            elapsed_days = np.array([t - start_t + 1.0 for t in batch_times])
        else:
            elapsed_days = np.arange(1.0, num_days + 1.0)

    def get_latest_state(bee_states, up_to_day_idx):
        for k in range(up_to_day_idx, -1, -1):
            state = str(bee_states[k]).strip().upper()
            if state in ['A', 'B', 'D']:
                return state
        return 'A'

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

    rows = []
    for bc in bee_cols:
        b_states = df_copy[bc].fillna("A").tolist()
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
            dur = float(elapsed_days[d_star]) if d_star < len(elapsed_days) else float(elapsed_days[-1])
            ev = 1
            alive_cons = cons_vals[:d_star+1] if cons_vals else []
        else:
            dur = float(elapsed_days[-1]) if len(elapsed_days) > 0 else 1.0
            ev = 0
            alive_cons = cons_vals

        mean_cons = float(np.mean(alive_cons)) if len(alive_cons) > 0 else 0.0
        total_cons = float(np.sum(alive_cons)) if len(alive_cons) > 0 else 0.0

        treatment_val = batch_name
        if 'Treatment' in df_copy.columns and pd.notna(df_copy['Treatment'].iloc[0]):
            treatment_val = str(df_copy['Treatment'].iloc[0]).strip()

        rows.append({
            'bee_id': bc,
            'cohort': batch_name,
            'treatment': treatment_val,
            'duration': max(dur, 0.5),
            'event': ev,
            'mean_consumption': mean_cons,
            'total_consumption': total_cons,
            'temp': temp_val if not np.isnan(temp_val) else 25.0,
            'humidity': humid_val if not np.isnan(humid_val) else 60.0,
        })

    return pd.DataFrame(rows)
