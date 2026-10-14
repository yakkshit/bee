"""
Core Time Parser Module.
Provides continuous elapsed day calculations from observation headers (Day 1, Day 1_2, Day 2, etc.)
and default experiment observation sheet templates.
Strictly enforces the Post-2h Starvation Rule (Excluding Time Zero / d0).
"""
import re
import numpy as np
import pandas as pd

DEFAULT_CSV = """Day,Date/time,Bee 1,Bee 2,Bee 3,Bee 4,Bee 5,Bee 6,Bee 7,Bee 8,Bee 9,Bee 10,Bee 11,Bee 12,Bee 13,Bee 14,Temp (°C),Humidity,Treatment,Observations/notes
Day 1,02.09.2026-16:00,A,D,A,A,A,A,A,A,D,A,A,A,A,D,23.3,55,sucrose solution 2 microliter,
Day 2,03.09.2026-16:00,A,D,A,A,A,A,A,A,D,A,A,A,A,D,24.4,45,sucrose solution 2 microliter,
Day 3,04.09.2026-16:00,A,D,A,A,B,A,A,A,D,A,A,A,A,D,24.7,58,sucrose solution 2 microliter,
Day 4,05.09.2026-16:00,A,D,A,A,B,A,A,A,D,A,A,A,A,D,23.8,61,sucrose solution 2 microliter,
Day 5,06.09.2026-16:00,A,D,D,A,B,A,A,A,D,A,A,D,A,D,23.7,51,sucrose solution 2 microliter,
Day 6,07.09.2026-16:00,D,D,D,A,B,A,A,A,D,A,A,D,A,D,24.5,64,sucrose solution 2 microliter,
Day 7,08.09.2026-16:00,D,D,D,B,B,A,A,A,D,A,A,D,A,D,24.8,62,sucrose solution 2 microliter,
Day 8,09.09.2026-16:00,D,D,D,B,B,A,A,A,D,A,A,D,A,D,23.4,59,sucrose solution 2 microliter,"""


def is_time_zero(t_val):
    """Checks if a day/time string represents Time Zero / pre-starvation baseline."""
    s = str(t_val).strip().lower()
    return s in ['d0', 'day 0', 'dy 0', 'd_0', '0', 'day0', 't0', 't_0', 'time 0', 'time0']


def filter_post_starvation_data(df):
    """
    Bioassay Protocol Enforcement:
    Excludes time zero / pre-starvation baseline observations (e.g. 'd0', 'Day 0').
    All statistical and kinetic calculations strictly start post-2h starvation period.
    """
    if df is None or df.empty:
        return df

    time_col = 'Day' if 'Day' in df.columns else ('Date/time' if 'Date/time' in df.columns else None)
    if time_col:
        mask = ~df[time_col].apply(is_time_zero)
        if not mask.all():
            return df[mask].copy().reset_index(drop=True)
    return df


def parse_time_data(df):
    """
    Parses the 'Day' column into continuous elapsed days starting post-2h starvation.
    Automatically filters out time zero (d0 / Day 0).
    """
    cleaned_df = filter_post_starvation_data(df)
    time_col_name = 'Day' if 'Day' in cleaned_df.columns else ('Date/time' if 'Date/time' in cleaned_df.columns else None)

    if time_col_name:
        raw_times = cleaned_df[time_col_name].astype(str).tolist()
    else:
        raw_times = [f"Day {i+1}" for i in range(len(cleaned_df))]

    def parse_to_days(t_str):
        t_str = str(t_str).strip()
        match = re.match(r"(?:Day|Dy|D)?\s*(\d+)(?:_(\d+)(?::(\d+))?)?", t_str, re.IGNORECASE)
        if match and match.group(1):
            d = int(match.group(1))
            # Treat single-digit observation indices without minutes (e.g. Day 1_2) as h = 0
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
