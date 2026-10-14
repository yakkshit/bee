"""
Core Individual Bee Survival Extractor Module.
Extracts subject-level duration, event censoring status, consumption,
and environmental conditions for Kaplan-Meier, Log-Rank, and Cox Proportional Hazards modeling.
Strictly adheres to the Resurrected Bee observation correction rule.
"""
import numpy as np
import pandas as pd

from modules.core.time_parser import parse_time_data


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
    if 'bee_id' in df_copy.columns:
        from modules.core.batch_parser import standardize_observation_df
        df_copy = standardize_observation_df(df_copy, default_treatment=batch_name)

    bee_cols = [c for c in df_copy.columns if str(c).lower().startswith('bee') and str(c).lower() != 'bee_id']
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
            alive_cons = cons_vals[:d_star + 1] if cons_vals else []
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
