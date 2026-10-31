"""
Core Individual Bee Survival Extractor Module.
Extracts subject-level duration, event censoring status, consumption,
and environmental conditions for Kaplan-Meier, Log-Rank, and Cox Proportional Hazards modeling.
Strictly adheres to the Resurrected Bee observation correction rule.
"""
import numpy as np
import pandas as pd

from modules.core.time_parser import parse_time_data
from modules.core.batch_parser import (
    get_bee_columns,
    get_incubator_temp_col,
    get_incubator_humidity_col,
    find_matching_cons_col
)

try:
    import streamlit as st
    cache_data_fn = st.cache_data
except Exception:
    def cache_data_fn(func):
        return func


@cache_data_fn
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

    bee_cols = get_bee_columns(df_copy)
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

    # Incubator environmental conditions (strictly ignoring outdoor/outside columns)
    incubator_temp_col = get_incubator_temp_col(df_copy.columns)
    incubator_humid_col = get_incubator_humidity_col(df_copy.columns)
    temp_val = np.nan
    humid_val = np.nan
    if incubator_temp_col:
        try:
            temp_val = float(pd.to_numeric(df_copy[incubator_temp_col], errors='coerce').mean())
        except Exception:
            pass
    if incubator_humid_col:
        try:
            humid_val = float(pd.to_numeric(df_copy[incubator_humid_col], errors='coerce').mean())
        except Exception:
            pass

    rows = []
    for bc in bee_cols:
        b_states = df_copy[bc].fillna("A").tolist()
        final_state = get_latest_state(b_states, num_days - 1)

        c_matched = find_matching_cons_col(df_copy.columns, bc)
        cons_vals = []
        if c_matched:
            cons_vals = pd.to_numeric(df_copy[c_matched], errors='coerce').fillna(0.0).tolist()

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

        # Feeding cessation endpoint: find when bee permanently stops eating (Cons <= 0)
        if cons_vals and any(v > 0 for v in cons_vals):
            pos_indices = [k for k, v in enumerate(cons_vals) if v > 0]
            last_feed_idx = max(pos_indices)
            if last_feed_idx < num_days - 1:
                cess_idx = min(last_feed_idx + 1, num_days - 1)
                dur_feed = float(elapsed_days[cess_idx]) if cess_idx < len(elapsed_days) else float(elapsed_days[-1])
                ev_feed = 1
            else:
                dur_feed = float(elapsed_days[-1]) if len(elapsed_days) > 0 else 1.0
                ev_feed = 0
        else:
            dur_feed = float(elapsed_days[0]) if len(elapsed_days) > 0 else 1.0
            ev_feed = 1 if final_state == 'D' else 0

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
            'feeding_cessation_duration': max(dur_feed, 0.5),
            'feeding_cessation_event': ev_feed,
            'temp': temp_val if not np.isnan(temp_val) else 25.0,
            'humidity': humid_val if not np.isnan(humid_val) else 60.0,
        })

    return pd.DataFrame(rows)
