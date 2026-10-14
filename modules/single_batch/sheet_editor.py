"""
Observation Sheet Editor and HTML Print-to-PDF Report Generator.
Handles dynamic cell editing, adding bee columns, resetting session state,
and exporting printable observation sheets.
Enforces the Post-2h Starvation Rule (Excluding d0 / Time Zero).
"""
import io
import os
import pandas as pd
import streamlit as st

from modules.core.time_parser import filter_post_starvation_data, is_time_zero, DEFAULT_CSV
from modules.parser import (
    standardize_observation_df,
    clean_cohort_name
)

CACHE_FILE = ".cached_df.csv"


def generate_html_report(dataframe, metadata):
    """Generates standard printable HTML observation sheet."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{ font-family: Arial, sans-serif; font-size: 11px; }}
        h1 {{ text-align: center; color: #1a2a5a; text-transform: uppercase; font-size: 18px; margin-bottom: 2px; }}
        .subtitle {{ text-align: center; font-style: italic; color: #555; margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #9fb4cc; padding: 6px; text-align: center; }}
        th {{ background-color: #e5eff8; color: #1a2a5a; font-weight: bold; }}
        .obs-table th {{ background-color: #1a3b5c; color: white; }}
        .meta-label {{ background-color: #e5eff8; font-weight: bold; color: #1a2a5a; text-align: left; width: 12%; }}
        .meta-val {{ text-align: left; width: 13%; font-style: italic; }}
        @media print {{
            @page {{ size: landscape; margin: 10mm; }}
            body {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
        }}
    </style>
    </head>
    <body>
        <h1>Honeybee Experiment Observation Sheet</h1>
        <div class="subtitle">Enter the study details, then record one observation per bee for each day.</div>

        <table>
            <tr>
                <td class="meta-label">Date</td><td class="meta-val">{metadata.get('meta_Date', '')}</td>
                <td class="meta-label">Start time</td><td class="meta-val">{metadata.get('meta_Start time', '')}</td>
                <td class="meta-label">Weather</td><td class="meta-val">{metadata.get('meta_Weather', '')}</td>
                <td class="meta-label">Observer</td><td class="meta-val">{metadata.get('meta_Observer', '')}</td>
            </tr>
            <tr>
                <td class="meta-label">Treatment</td><td class="meta-val">{metadata.get('meta_Treatment', '')}</td>
                <td class="meta-label">Hive / colony ID</td><td class="meta-val">{metadata.get('meta_Hive / colony ID', '')}</td>
                <td class="meta-label">Bee type</td><td class="meta-val">{metadata.get('meta_Bee type', '')}</td>
                <td class="meta-label">Feeder / resource</td><td class="meta-val">{metadata.get('meta_Feeder / resource', '')}</td>
            </tr>
            <tr>
                <td class="meta-label">Purpose / motivation</td><td class="meta-val">{metadata.get('meta_Purpose / motivation', '')}</td>
                <td class="meta-label">Outside temp(°C)</td><td class="meta-val">{metadata.get('meta_Outside temp(°C)', '')}</td>
                <td class="meta-label">Outdoor humidity</td><td class="meta-val">{metadata.get('meta_Outdoor humidity', '')}</td>
                <td class="meta-label">Other conditions</td><td class="meta-val">{metadata.get('meta_Other conditions', '')}</td>
            </tr>
        </table>

        <h3 style="color: #1a2a5a; margin-bottom: 5px; font-size: 14px;">DAILY INDIVIDUAL OBSERVATIONS (Post-2h Starvation)</h3>
        <table class="obs-table">
            <tr>
    """
    for col in dataframe.columns:
        html += f"<th>{col}</th>\n"
    html += "</tr>\n"

    for _, row in dataframe.iterrows():
        html += "<tr>\n"
        for val in row:
            v = val if pd.notna(val) else ''
            html += f"<td>{v}</td>\n"
        html += "</tr>\n"

    html += """
        </table>

        <table style="border: none; margin-top: 10px;">
            <tr>
                <td style="text-align: left; border: 1px solid #9fb4cc; background: #f5f8fc;"><b>Bee status:</b> A = alive, antennae moving &nbsp;|&nbsp; B = alive, antennae not moving &nbsp;|&nbsp; D = dead</td>
                <td style="text-align: left; border: 1px solid #9fb4cc; background: #f5f8fc;"><b>Score key:</b> -, 0, +, ++ &nbsp;&nbsp;(define the meaning for your experimental protocol)</td>
            </tr>
            <tr>
                <td style="text-align: left; border: 1px solid #9fb4cc; background: #f5f8fc;"><b>Additional note:</b> Record behaviour, feeding, movement and unusual events.</td>
                <td style="text-align: left; border: 1px solid #9fb4cc; background: #f5f8fc;"><b>End time:</b> <i>Click or type here</i></td>
            </tr>
        </table>
    </body>
    </html>
    """
    return html


def render_observation_sheet_editor():
    """Renders the data editor, file upload, add bee columns, and export buttons."""
    if 'df' not in st.session_state:
        if os.path.exists(CACHE_FILE):
            try:
                st.session_state.df = pd.read_csv(CACHE_FILE)
            except Exception:
                st.session_state.df = pd.read_csv(io.StringIO(DEFAULT_CSV))
        else:
            st.session_state.df = pd.read_csv(io.StringIO(DEFAULT_CSV))

        # Filter out pre-starvation time zero if present
        st.session_state.df = filter_post_starvation_data(st.session_state.df)

        bee_cols_init = [c for c in st.session_state.df.columns if str(c).lower().startswith('bee')]
        for b in bee_cols_init:
            cons_col = b.replace("Bee ", "Cons ") + " (µL)"
            if cons_col not in st.session_state.df.columns:
                st.session_state.df.insert(st.session_state.df.columns.get_loc(b) + 1, cons_col, "0.0")

        st.session_state.df = st.session_state.df.astype(str)
        st.session_state.df.replace("nan", "", inplace=True)

    st.write("### Daily Individual Observations")
    st.info("🕒 **Bioassay Protocol Active**: Starting observations are strictly post-2h starvation period (Time Zero / baseline d0 is excluded from kinetics and hazard models).")

    if 'uploaded_file_dict' not in st.session_state:
        st.session_state['uploaded_file_dict'] = {}

    uploaded_files = st.file_uploader(
        "Upload CSV Observation File(s) (Multiple files allowed)",
        type=["csv"],
        accept_multiple_files=True,
        key="single_csv_uploader",
        help="Upload one or more observation CSV files. All uploaded files are automatically shared and analyzed across all tabs."
    )

    if uploaded_files:
        files_updated = False
        for f in uploaded_files:
            if f.name not in st.session_state['uploaded_file_dict']:
                try:
                    raw_f_df = pd.read_csv(f, sep=None, engine='python')
                    clean_f_df = standardize_observation_df(raw_f_df)
                    clean_f_df = clean_f_df.astype(str)
                    clean_f_df.replace("nan", "", inplace=True)
                    norm_label = clean_cohort_name(f.name)
                    st.session_state['uploaded_file_dict'][f.name] = (norm_label, clean_f_df)
                    files_updated = True
                except Exception as ex:
                    st.error(f"Error reading {f.name}: {ex}")

        if files_updated:
            # Set active sheet to the first uploaded file
            first_key = list(st.session_state['uploaded_file_dict'].keys())[0]
            st.session_state.df = st.session_state['uploaded_file_dict'][first_key][1].copy()
            st.rerun()

    # If multiple files are uploaded, display a selector to choose the active sheet in this table editor
    if len(st.session_state.get('uploaded_file_dict', {})) > 1:
        f_keys = list(st.session_state['uploaded_file_dict'].keys())
        active_f = st.selectbox(
            "📂 Active Cohort Sheet to View / Edit in Table:",
            f_keys,
            format_func=lambda k: f"{st.session_state['uploaded_file_dict'][k][0]} ({k})",
            key="tab1_active_sheet_selector"
        )
        if active_f:
            st.session_state.df = st.session_state['uploaded_file_dict'][active_f][1]

    col1, col2, col3 = st.columns([1.5, 1, 4])
    with col1:
        if st.button("➕ Add Bee Column"):
            bee_cols = [c for c in st.session_state.df.columns if str(c).lower().startswith('bee')]
            new_idx = len(bee_cols) + 1
            new_col_name = f"Bee {new_idx}"
            new_cons_name = f"Cons {new_idx} (µL)"
            try:
                temp_idx = list(st.session_state.df.columns).index("Temp (°C)")
            except ValueError:
                temp_idx = len(st.session_state.df.columns)
            st.session_state.df.insert(temp_idx, new_col_name, "A")
            st.session_state.df.insert(temp_idx + 1, new_cons_name, "0.0")
            st.rerun()
    with col2:
        if st.button("🔄 Reset Data / Uploads"):
            if 'df' in st.session_state:
                del st.session_state['df']
            if 'uploaded_file_dict' in st.session_state:
                st.session_state['uploaded_file_dict'] = {}
            if 'processed_batches' in st.session_state:
                del st.session_state['processed_batches']
            if os.path.exists(CACHE_FILE):
                os.remove(CACHE_FILE)
            st.rerun()
    with col3:
        st.markdown("You can edit the cells below. **To add a new Day (row)**, scroll to the bottom of the table and click the empty row.")

    edited_df = st.data_editor(st.session_state.df, num_rows="dynamic", use_container_width=True)
    st.session_state.df = edited_df
    df = edited_df.copy()

    # Sync back to central uploaded file dictionary if an uploaded file is active
    if 'uploaded_file_dict' in st.session_state and st.session_state['uploaded_file_dict']:
        cur_active_key = locals().get('active_f') or list(st.session_state['uploaded_file_dict'].keys())[0]
        if cur_active_key in st.session_state['uploaded_file_dict']:
            cur_lbl = st.session_state['uploaded_file_dict'][cur_active_key][0]
            st.session_state['uploaded_file_dict'][cur_active_key] = (cur_lbl, df)

    try:
        df.to_csv(CACHE_FILE, index=False)
    except Exception:
        pass


    st.markdown("---")
    st.header("🖨️ Observation Sheet Export")
    st.markdown("Download a printable version of your observation sheet (matching the template). Open in browser and press **Ctrl+P / Cmd+P** to save as landscape PDF.")

    html_content = generate_html_report(df, st.session_state)
    st.download_button(
        label="📄 Download Observation Sheet (HTML format for Print-to-PDF)",
        data=html_content,
        file_name="Observation_Sheet.html",
        mime="text/html",
        key="dl_obs_sheet_html"
    )

    return df
