"""
CSV Health Auditor & Bioassay Diagnostic Module.
Audits uploaded CSV datasets for formatting anomalies, missing observation headers,
non-numeric environmental entries, and bioassay protocol boundary conditions.
Surfaces non-blocking, actionable warnings at the top of the application.
"""
import re
import numpy as np
import pandas as pd
import streamlit as st


def clean_numeric_value(val):
    """Parses a numerical value, auto-converting Excel time strings like '24:00:00' to float."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if not s or s.lower() in ['nan', 'none', 'null', '-', '']:
        return np.nan
    if ':' in s:
        parts = s.split(':')
        try:
            return float(parts[0]) + (float(parts[1]) / 60.0 if len(parts) > 1 else 0.0)
        except Exception:
            pass
    try:
        return float(s)
    except Exception:
        return np.nan


def audit_single_dataframe(df, filename="Active Dataset"):
    """
    Audits a single CSV DataFrame for syntax errors, missing data, and bioassay anomalies.
    Returns a list of dicts: [{'type': 'error'|'warning'|'info', 'file': filename, 'msg': str}]
    """
    diagnostics = []
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return diagnostics

    # 1. Check Day column missing / NaN entries
    day_col = 'Day' if 'Day' in df.columns else ('Date/time' if 'Date/time' in df.columns else None)
    if day_col:
        for idx, val in enumerate(df[day_col]):
            s_val = str(val).strip()
            if not s_val or s_val.lower() in ['nan', 'none', 'null']:
                diagnostics.append({
                    'type': 'warning',
                    'file': filename,
                    'msg': f"Row {idx + 1}: Missing '{day_col}' header label. System automatically inferred and repaired it as 'Day {idx + 1}'."
                })

    # 2. Check Incubator Temperature & Humidity format
    temp_cols = [c for c in df.columns if 'temp' in str(c).lower() and 'out' not in str(c).lower() and 'treat' not in str(c).lower()]
    for tc in temp_cols:
        for idx, val in enumerate(df[tc]):
            if pd.isna(val):
                continue
            s_val = str(val).strip()
            if ':' in s_val:
                repaired_num = clean_numeric_value(s_val)
                diagnostics.append({
                    'type': 'warning',
                    'file': filename,
                    'msg': f"Row {idx + 1} [{tc}]: Time-formatted string '{s_val}' detected (likely Excel auto-format). Auto-converted to {repaired_num:.1f}°C."
                })

    # 3. Check Individual Bee status integrity
    bee_cols = [c for c in df.columns if re.match(r'^bee[\s_#]*\d+$', str(c).strip(), re.IGNORECASE)]
    for bc in bee_cols:
        for idx, val in enumerate(df[bc]):
            if pd.isna(val):
                continue
            v_str = str(val).strip().upper()
            if v_str not in ['A', 'B', 'D', '']:
                diagnostics.append({
                    'type': 'error',
                    'file': filename,
                    'msg': f"Row {idx + 1} [{bc}]: Unrecognized bee health status '{val}'. Expected 'A' (Alive), 'B' (Affected/Moribund), or 'D' (Dead)."
                })

    # 4. Bioassay Protocol: Mortality rate & LT50 reachable check
    if bee_cols and len(df) > 0:
        total_bees = len(bee_cols)
        final_row = df.iloc[-1]
        dead_final = sum(1 for bc in bee_cols if str(final_row[bc]).strip().upper() == 'D')
        final_mort = (dead_final / total_bees) * 100.0 if total_bees > 0 else 0.0

        if final_mort < 20.0 and len(df) >= 3:
            diagnostics.append({
                'type': 'info',
                'file': filename,
                'msg': f"Low cumulative mortality ({final_mort:.1f}% across {len(df)} days; {dead_final}/{total_bees} bees). Lethal median time (LT50) was not reached during this trial."
            })

    return diagnostics


def audit_batch_collection(raw_batch_dict):
    """
    Audits a dictionary of {filename: df} batches and checks for cross-batch duration differences.
    """
    all_diagnostics = []
    durations = {}

    for fname, df in raw_batch_dict.items():
        diags = audit_single_dataframe(df, fname)
        all_diagnostics.extend(diags)
        if df is not None and isinstance(df, pd.DataFrame):
            durations[fname] = len(df)

    # Check for duration discrepancies across files
    if len(durations) > 1:
        unique_durations = set(durations.values())
        if len(unique_durations) > 1:
            dur_breakdown = ", ".join([f"**{f}** ({d}d)" for f, d in durations.items()])
            all_diagnostics.append({
                'type': 'info',
                'file': 'Multiple Datasets',
                'msg': f"Observation trial lengths differ across uploaded CSV files: {dur_breakdown}."
            })

    return all_diagnostics


def render_csv_diagnostics_banner(diagnostics, container_key="global_csv_audit"):
    """
    Renders an alert banner in Streamlit displaying any data anomalies or bioassay warnings.
    """
    if not diagnostics:
        return

    errors = [d for d in diagnostics if d['type'] == 'error']
    warnings = [d for d in diagnostics if d['type'] == 'warning']
    infos = [d for d in diagnostics if d['type'] == 'info']

    total_issues = len(errors) + len(warnings) + len(infos)
    banner_title = f"⚠️ CSV Health & Bioassay Protocol Notices ({total_issues} item{'s' if total_issues != 1 else ''})"

    with st.expander(banner_title, expanded=(len(errors) > 0 or len(warnings) > 0)):
        if errors:
            st.markdown("##### 🚨 Critical Formatting Errors:")
            for e in errors:
                st.error(f"**[{e['file']}]** {e['msg']}")

        if warnings:
            st.markdown("##### ⚠️ Data Quality Auto-Repairs & Warnings:")
            for w in warnings:
                st.warning(f"**[{w['file']}]** {w['msg']}")

        if infos:
            st.markdown("##### ℹ️ Bioassay & Trial Duration Notices:")
            for inf in infos:
                st.info(f"**[{inf['file']}]** {inf['msg']}")

