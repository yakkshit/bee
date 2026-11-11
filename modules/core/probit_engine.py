"""
Core Finney Probit Mathematical Engine.
Computes theoretical Probit-derived lethal time thresholds (LT10, LT20, LT30, LT50, LT90)
and observed empirical bioassay transition days.
"""
import numpy as np
from scipy.stats import norm

try:
    import streamlit as st
    cache_data_fn = st.cache_data
except Exception:
    def cache_data_fn(func=None, **kwargs):
        if func is not None:
            return func
        return lambda f: f


@cache_data_fn(show_spinner=False)
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

        # Theoretical Probit OLS estimate (with overflow guard)
        if slope_b > 1e-6:
            exponent = (probit_score - intercept_a) / slope_b
            if exponent > 4.0:  # > 10,000 days, well beyond acute bioassay horizon
                theo_lt = np.nan
                theo_elapsed = np.nan
                theo_elapsed_int = np.nan
            else:
                try:
                    theo_lt = float(10.0 ** exponent)
                except OverflowError:
                    theo_lt = np.nan

                if np.isfinite(theo_lt) and theo_lt > 0:
                    if theo_lt <= start_time:
                        theo_elapsed = max(0.1, theo_lt)
                        theo_elapsed_int = 1
                    else:
                        theo_elapsed = theo_lt - start_time
                        theo_elapsed_int = max(1, int(np.ceil(theo_elapsed)))
                else:
                    theo_lt = np.nan
                    theo_elapsed = np.nan
                    theo_elapsed_int = np.nan
        else:
            theo_lt = np.nan
            theo_elapsed = np.nan
            theo_elapsed_int = np.nan

        # Empirical bioassay observed estimate
        emp_lt = np.nan
        emp_elapsed = np.nan
        emp_elapsed_int = np.nan
        reached = False

        max_trial_time = float(times_numeric[-1]) if (times_numeric is not None and len(times_numeric) > 0 and np.isfinite(times_numeric[-1])) else 6.0
        max_obs_mort = float(max(mortality_pct)) if (mortality_pct is not None and len(mortality_pct) > 0) else 0.0

        if times_numeric is not None and mortality_pct is not None and len(times_numeric) > 0 and len(mortality_pct) > 0:
            t_arr = np.array(times_numeric, dtype=float)
            m_arr = np.array(mortality_pct, dtype=float)

            if m_arr[0] >= p:
                t_first = float(t_arr[0])
                if m_arr[0] > 0:
                    emp_lt = max(0.1, (p / m_arr[0]) * t_first)
                else:
                    emp_lt = t_first
                emp_elapsed = max(0.1, emp_lt - max(0.0, start_time - 1.0))
                emp_elapsed_int = 1
                reached = True
            else:
                for idx in range(1, len(m_arr)):
                    if m_arr[idx] >= p:
                        t_prev = float(t_arr[idx - 1])
                        t_curr = float(t_arr[idx])
                        m_prev = float(m_arr[idx - 1])
                        m_curr = float(m_arr[idx])

                        if np.isfinite(t_prev) and np.isfinite(t_curr):
                            if m_curr > m_prev:
                                emp_lt = t_prev + ((p - m_prev) / (m_curr - m_prev)) * (t_curr - t_prev)
                            else:
                                emp_lt = t_curr

                            emp_elapsed = max(0.1, emp_lt - start_time)
                            emp_elapsed_int = max(1, int(np.ceil(emp_elapsed)))
                            reached = True
                            break

        # Effective elapsed days: prefer empirical observation if reached
        # Fallback to theoretical model ONLY if interpolation/extrapolation falls strictly within the actual trial duration
        if reached and np.isfinite(emp_elapsed_int):
            effective_elapsed_int = emp_elapsed_int
            status_desc = f"Observed ({emp_elapsed:.1f}d)"
        elif np.isfinite(theo_elapsed_int) and theo_elapsed <= max_trial_time and max_obs_mort >= (p * 0.5):
            effective_elapsed_int = theo_elapsed_int
            status_desc = f"Probit Model ({theo_elapsed:.1f}d)"
        else:
            effective_elapsed_int = np.nan
            if max_obs_mort < p:
                status_desc = f"> {int(round(max_trial_time))}d (Not Reached)"
            else:
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
