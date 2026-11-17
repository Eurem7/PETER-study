"""PETER Study — Streamlit Web Interface
Run:  streamlit run peter_app.py

Place the accompanying config at  .streamlit/config.toml  next to this file.
"""

import os, sys, pathlib, importlib.util, re, io, contextlib, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image as PILImage
from cycler import cycler

st.set_page_config(
    page_title="PETER Study",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────── design tokens
BRAND  = "#14452f"
MOSS   = "#3f7d58"
SAGE   = "#d9e7de"
INK    = "#16201b"
SLATE  = "#5b6760"
LINE   = "#dfe4e0"

# ─────────────────────────────────────────────── CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {
    --brand:      #14452f;
    --brand-deep: #0e3321;
    --moss:       #3f7d58;
    --sage:       #d9e7de;
    --sage-soft:  #eef4f0;
    --bg:         #eef1ee;
    --paper:      #ffffff;
    --ink:        #16201b;
    --slate:      #5b6760;
    --faint:      #8c978f;
    --line:       #dfe4e0;
    --line-strong:#c9d1cb;
    --amber-bg:   #fff7e6;
    --amber-line: #f0d9a4;
    --amber-ink:  #6b4710;
    --blue-bg:    #edf4fb;
    --blue-line:  #c3d9ee;
    --blue-ink:   #1d4a73;
}

/* ── Base ── */
.stApp {
    background: var(--bg);
    color: var(--ink);
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.stApp button, .stApp input, .stApp textarea, .stApp select {
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    font-size: 0.92rem;
    line-height: 1.65;
    color: #334039;
}
[data-testid="stCaptionContainer"] { color: var(--slate); }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] { display: none !important; }

/* ── Page container ── */
.block-container {
    max-width: 1180px !important;
    padding: 18px 28px 72px !important;
}

/* ── Top navigation ── */
.st-key-topnav {
    position: sticky;
    top: 12px;
    z-index: 999;
    background: rgba(255, 255, 255, 0.94);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 8px 16px;
    margin-bottom: 32px;
}
.brand { display: flex; align-items: center; gap: 11px; }
.brand-mark {
    width: 34px; height: 34px;
    border-radius: 8px;
    background: var(--brand);
    display: flex; align-items: center; justify-content: center;
    font-size: 17px;
    flex-shrink: 0;
}
.brand-name { font-size: 0.95rem; font-weight: 700; color: var(--brand); line-height: 1.15; letter-spacing: -0.01em; }
.brand-sub  { font-size: 0.72rem; color: var(--slate); line-height: 1.2; }
.nav-version { text-align: right; font-size: 0.72rem; color: var(--faint); }

/* native st.radio restyled as tab strip */
.st-key-topnav [data-testid="stRadio"] > label { display: none; }
.st-key-topnav [role="radiogroup"] { gap: 4px; flex-wrap: nowrap; justify-content: center; }
.st-key-topnav [role="radiogroup"] > label {
    margin: 0;
    padding: 7px 16px;
    border-radius: 7px;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: background 0.15s;
}
.st-key-topnav [role="radiogroup"] > label > div:first-child { display: none; }
.st-key-topnav [role="radiogroup"] > label p {
    font-size: 0.86rem;
    font-weight: 500;
    color: var(--slate);
    white-space: nowrap;
}
.st-key-topnav [role="radiogroup"] > label:hover { background: var(--sage-soft); }
.st-key-topnav [role="radiogroup"] > label:hover p { color: var(--ink); }
.st-key-topnav [role="radiogroup"] > label:has(input:checked) {
    background: var(--sage-soft);
    border-bottom-color: var(--brand);
    border-radius: 7px 7px 0 0;
}
.st-key-topnav [role="radiogroup"] > label:has(input:checked) p {
    color: var(--brand);
    font-weight: 600;
}

/* ── Page header ── */
.page-header { margin: 0 0 28px; }
.breadcrumb { font-size: 0.8rem; color: var(--faint); margin-bottom: 8px; }
.breadcrumb .sep { margin: 0 7px; color: var(--line-strong); }
.stApp .page-header h1 {
    font-size: 1.85rem;
    font-weight: 700;
    color: var(--ink);
    letter-spacing: -0.025em;
    line-height: 1.2;
    margin: 0 0 8px;
    padding: 0;
}
.stApp .page-header p {
    font-size: 0.95rem;
    color: var(--slate);
    line-height: 1.6;
    max-width: 68ch;
    margin: 0;
}

/* ── Panels ── */
[class*="st-key-panel_"] {
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 22px 24px 20px;
    margin-bottom: 18px;
}
.panel-head { margin-bottom: 14px; }
.panel-title {
    font-size: 1.02rem;
    font-weight: 600;
    color: var(--ink);
    letter-spacing: -0.01em;
    padding-left: 11px;
    border-left: 3px solid var(--moss);
    line-height: 1.3;
}
.panel-desc {
    font-size: 0.85rem;
    color: var(--slate);
    line-height: 1.6;
    margin-top: 6px;
    padding-left: 14px;
    max-width: 80ch;
}

/* ── Stat band ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 8px;
    margin-bottom: 20px;
    overflow: hidden;
}
.stat-box {
    padding: 16px 20px;
    border-right: 1px solid var(--line);
}
.stat-box:last-child { border-right: none; }
.stat-box .s-label { font-size: 0.78rem; color: var(--slate); margin-bottom: 4px; }
.stat-box .s-val {
    font-size: 1.7rem;
    font-weight: 600;
    color: var(--brand);
    line-height: 1.1;
    letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums;
}
.stat-box .s-sub { font-size: 0.74rem; color: var(--faint); margin-top: 3px; }

/* ── Column mapping table ── */
.col-map-table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
.col-map-table th {
    background: var(--sage-soft);
    color: var(--brand);
    font-weight: 600;
    text-align: left;
    padding: 8px 12px;
    border-bottom: 2px solid var(--line-strong);
}
.col-map-table td { padding: 7px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }
.col-map-table tr:last-child td { border-bottom: none; }
.col-map-table .canon { font-family: 'IBM Plex Mono', monospace; color: var(--ink); font-weight: 500; }
.col-map-table .accept { color: var(--slate); font-size: 0.78rem; line-height: 1.6; }
.col-map-table .req { color: var(--brand); font-size: 0.72rem; font-weight: 600; }
.col-map-table .opt { color: var(--faint);  font-size: 0.72rem; }

/* ── Step list (empty state) ── */
.step-callout {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 14px 0;
    border-bottom: 1px solid var(--line);
}
.step-callout:last-child { border-bottom: none; }
.step-callout .sc-num {
    width: 26px; height: 26px;
    border-radius: 50%;
    background: var(--sage);
    color: var(--brand);
    font-size: 0.78rem;
    font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.sc-title { font-size: 0.92rem; font-weight: 600; color: var(--ink); }
.sc-text  { font-size: 0.84rem; color: var(--slate); line-height: 1.5; margin-top: 1px; }

/* ── Callouts ── */
.callout {
    border-radius: 6px;
    padding: 11px 15px;
    font-size: 0.85rem;
    line-height: 1.6;
    border: 1px solid;
    border-left-width: 3px;
}
.callout.note { background: var(--amber-bg); border-color: var(--amber-line); color: var(--amber-ink); }
.callout.info { background: var(--blue-bg);  border-color: var(--blue-line);  color: var(--blue-ink); }

/* ── Buttons ── */
.stApp [data-testid="stBaseButton-primary"],
.stApp button[kind="primary"] {
    background: var(--brand);
    border: 1px solid var(--brand);
    color: #fff;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.86rem;
    padding: 0.5rem 1.3rem;
    transition: background 0.15s, border-color 0.15s;
}
.stApp [data-testid="stBaseButton-primary"]:hover,
.stApp button[kind="primary"]:hover {
    background: var(--brand-deep);
    border-color: var(--brand-deep);
    color: #fff;
}
.stApp [data-testid="stBaseButton-primary"] p,
.stApp button[kind="primary"] p { color: #fff; font-size: 0.86rem; }

.stApp [data-testid="stBaseButton-secondary"],
.stApp [data-testid="stDownloadButton"] button {
    background: var(--paper);
    border: 1px solid var(--line-strong);
    color: var(--ink);
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.84rem;
    padding: 0.4rem 1rem;
}
.stApp [data-testid="stBaseButton-secondary"]:hover,
.stApp [data-testid="stDownloadButton"] button:hover {
    border-color: var(--moss);
    color: var(--brand);
    background: var(--sage-soft);
}

/* ── Inputs ── */
div[data-testid="stSelectbox"] label p,
div[data-testid="stSlider"] label p,
div[data-testid="stRadio"] label p { font-size: 0.84rem; font-weight: 500; color: var(--ink); }

[data-testid="stFileUploaderDropzone"] {
    background: var(--sage-soft);
    border: 1.5px dashed #9fc4ad;
    border-radius: 8px;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--moss); }

/* ── Misc components ── */
[data-testid="stExpander"] {
    border: 1px solid var(--line) !important;
    border-radius: 8px !important;
    background: var(--paper) !important;
    margin-top: 6px;
}
[data-testid="stExpander"] summary p { font-size: 0.88rem; font-weight: 500; }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
[data-testid="stCode"] pre { font-size: 0.78rem; line-height: 1.55; }
[data-testid="stCode"] { border: 1px solid var(--line); border-radius: 8px; }
[data-testid="stImage"] img { border: 1px solid var(--line); border-radius: 8px; background: #fff; }
[data-testid="stImageCaption"] { color: var(--slate); font-size: 0.78rem; }
[data-testid="stAlert"] { border-radius: 8px; font-size: 0.86rem; }

/* ── Narrow screens ── */
@media (max-width: 820px) {
    .block-container { padding: 12px 14px 56px !important; }
    .st-key-topnav { top: 6px; padding: 8px 10px; }
    .st-key-topnav [role="radiogroup"] { overflow-x: auto; justify-content: flex-start; }
    .nav-version, .brand-sub { display: none; }
    [class*="st-key-panel_"] { padding: 16px 14px; }
    .stApp .page-header h1 { font-size: 1.5rem; }
}
</style>
""", unsafe_allow_html=True)

# Matplotlib style for in-app charts
MPL_RC = {
    "font.family": "sans-serif",
    "font.size": 9.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#c9d1cb",
    "axes.labelcolor": "#334039",
    "axes.titlesize": 11,
    "axes.titleweight": "600",
    "axes.titlelocation": "left",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#eef1ee",
    "grid.linewidth": 0.9,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "legend.frameon": False,
    "axes.prop_cycle": cycler(color=[BRAND, "#d98a1f", "#3b7fb8", "#b8463b", "#7a5ca8", "#5c9c7a"]),
}


# ─────────────────────────────────────────────── load pa module
@st.cache_resource
def load_pa():
    here   = pathlib.Path(__file__).parent
    script = here / "peter_analysis.py"
    if not script.exists():
        st.error(f"peter_analysis.py not found in {here}."); st.stop()
    text = script.read_text(encoding="utf-8", errors="ignore")
    m    = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    ver  = m.group(1) if m else "unknown"
    sys.modules.pop("peter_analysis", None)
    spec = importlib.util.spec_from_file_location("peter_analysis", script)
    pa   = importlib.util.module_from_spec(spec)
    sys.modules["peter_analysis"] = pa
    spec.loader.exec_module(pa)
    import matplotlib; matplotlib.use("Agg")
    return pa, ver

pa, pa_ver = load_pa()

# ─────────────────────────────────────────────── session helpers
OUT = "results"
os.makedirs(OUT, exist_ok=True)

def _get(key, default=None): return st.session_state.get(key, default)
def _set(key, val):          st.session_state[key] = val

def _img(path, caption=""):
    if os.path.exists(path):
        st.image(PILImage.open(path), caption=caption, use_container_width=True)

def _dl(path, label="Download"):
    if os.path.exists(path):
        with open(path, "rb") as fh:
            st.download_button(label, fh.read(), file_name=os.path.basename(path))

def capture(func, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = func(*args, **kwargs)
    return result, buf.getvalue()

def go(page):
    st.session_state["nav"] = page

# ─────────────────────────────────────────────── UI helpers
def stat(label, value, sub=""):
    if isinstance(value, (int, np.integer)):
        value = f"{int(value):,}"
    return (f'<div class="stat-box"><div class="s-label">{label}</div>'
            f'<div class="s-val">{value}</div>'
            f'<div class="s-sub">{sub}</div></div>')

def step_note(num, title, text):
    st.markdown(
        f'<div class="step-callout">'
        f'<div class="sc-num">{num}</div>'
        f'<div><div class="sc-title">{title}</div>'
        f'<div class="sc-text">{text}</div></div></div>',
        unsafe_allow_html=True,
    )

def callout(kind, text):
    st.markdown(f'<div class="callout {kind}">{text}</div>', unsafe_allow_html=True)

def page_header(crumb, title, subtitle):
    st.markdown(
        f'<div class="page-header">'
        f'<div class="breadcrumb">PETER Study<span class="sep">›</span>{crumb}</div>'
        f'<h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )

def _keyed_container(key):
    try:
        return st.container(key=key)
    except TypeError:
        return st.container()

@contextlib.contextmanager
def panel(key, title, desc=""):
    with _keyed_container(f"panel_{key}"):
        desc_html = f'<div class="panel-desc">{desc}</div>' if desc else ""
        st.markdown(
            f'<div class="panel-head"><div class="panel-title">{title}</div>{desc_html}</div>',
            unsafe_allow_html=True,
        )
        yield


# ─────────────────────────────────────────────── top nav
data_ok = _get("wb_path") is not None
q1a_ok  = _get("wash_model") is not None
q2_ok   = _get("window_grid") is not None

NAV = [
    ("load", "Data",       data_ok),
    ("q1",   "Q1 Washing", q1a_ok),
    ("q2",   "Q2 Window",  q2_ok),
    ("q3",   "Q3 Carbon",  False),
]
NAV_LABEL = {k: (f"✓ {name}" if done else name) for k, name, done in NAV}

st.session_state.setdefault("nav", "load")

with _keyed_container("topnav"):
    c_brand, c_nav, c_ver = st.columns([2.3, 6, 1.2], vertical_alignment="center")
    with c_brand:
        st.markdown(
            '<div class="brand"><div class="brand-mark">🌲</div>'
            '<div><div class="brand-name">PETER Study</div>'
            '<div class="brand-sub">Analysis pipeline</div></div></div>',
            unsafe_allow_html=True,
        )
    with c_nav:
        page_key = st.radio(
            "Navigate", [k for k, _, _ in NAV],
            format_func=lambda k: NAV_LABEL[k],
            horizontal=True, key="nav", label_visibility="collapsed",
        )
    with c_ver:
        st.markdown(f'<div class="nav-version">v{pa_ver}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════ PAGE: LOAD DATA
if page_key == "load":
    page_header(
        "Data",
        "Load your workbook",
        "Upload PETER_Data_Workbook.xlsx to begin. "
        "Column names are matched automatically — minor differences in spelling or "
        "capitalisation are handled without any changes to your file.",
    )

    with panel("upload", "Workbook file", "Accepts .xlsx and .xlsm files."):
        uploaded = st.file_uploader(
            "Choose your workbook (.xlsx)", type=["xlsx", "xlsm"],
            label_visibility="collapsed",
        )
        if uploaded:
            wb_path = os.path.join(OUT, uploaded.name)
            with open(wb_path, "wb") as f: f.write(uploaded.read())
            try:
                pa.check_workbook(wb_path)
                _set("wb_path", wb_path)
                wash = pa.load_washing(wb_path)
                runs = pa.load_runs(wb_path)
                _set("wash", wash); _set("runs", runs)
                st.success(f"Loaded **{uploaded.name}**")
            except Exception as e:
                st.error(str(e))
                # Show the column guide automatically on error
                _set("show_col_guide", True)
                st.stop()

    wb = _get("wb_path")
    if wb:
        wash = _get("wash"); runs = _get("runs")

        st.markdown(
            '<div class="stat-row">'
            + stat("Wash runs",    len(wash))
            + stat("Md recorded",  int(wash["Md"].notna().sum()), "final dry mass")
            + stat("PETER rows",   len(runs))
            + stat("Washes",       int(runs["wash"].nunique()))
            + stat("Carbon doses", int(runs["Cpct"].nunique()))
            + '</div>',
            unsafe_allow_html=True,
        )

        with panel("previews", "Sheet previews"):
            c1, c2 = st.columns(2)
            with c1:
                st.caption("**Washing sheet**")
                st.dataframe(wash[["Run", "T", "t", "W", "Md", "SRY"]].round(2), height=260)
            with c2:
                st.caption("**PETER_runs sheet** (first 20 rows)")
                st.dataframe(runs[["wash", "T", "t", "W", "Cpct",
                                    "HCHO_net_peak", "TVOC_net_peak",
                                    "CO_net_peak", "CO2_net_peak"]].head(20).round(2), height=260)

        with panel("missing", "Missing values"):
            cols = ["Md"] + [c for c in runs.columns if c.endswith(("net_peak", "net_plateau"))]
            mv   = pd.concat([
                wash[["Md"]].isna().sum().rename("Washing"),
                runs[[c for c in cols[1:] if c in runs]].isna().sum().rename("PETER_runs"),
            ], axis=1).fillna("")
            st.dataframe(mv)
            callout("note",
                    "Empty cells in Md or any emission column are fine. "
                    "The analysis skips any response with fewer than 20 non-missing values.")

        st.button("Continue to Q1", type="primary", on_click=go, args=("q1",))

    else:
        with panel("steps", "How it works"):
            step_note(1, "Upload your workbook", "Use the file picker above to choose your .xlsx file.")
            step_note(2, "Inspect both sheets",  "Previews of the Washing and PETER_runs sheets appear here.")
            step_note(3, "Work through Q1 → Q3", "Use the navigation bar at the top to move through the analysis in order.")

    # ── Column name reference guide ──────────────────────────────────────
    # Shows automatically on a column-mismatch error; always available to expand.
    show_guide = _get("show_col_guide", False)
    with st.expander("📋 Sheet & column name reference — click to view accepted names", expanded=show_guide):
        st.markdown(
            "The pipeline matches both **sheet names** and **column names** automatically. "
            "If something cannot be matched it will be listed in the error above. "
            "Use this table to check the exact name or any accepted alternative.",
            unsafe_allow_html=False,
        )
        st.markdown("**Accepted sheet names**")
        sheet_rows = {
            "Washing":    "Wash, Washes, Washing runs, Washing data, Wash runs, Part 1",
            "PETER_runs": "PETER runs, PETER, Peter Run, Emission runs, Runs, Part 2, Chamber runs",
            "Controls":   "Control, Raw dust, Raw, Reference, Control runs  *(optional sheet)*",
        }
        sh_html = '<table class="col-map-table"><thead><tr><th>Canonical sheet name</th><th>Also accepted (examples)</th></tr></thead><tbody>'
        for canon, accepted in sheet_rows.items():
            sh_html += f"<tr><td class='canon'>{canon}</td><td class='accept'>{accepted}</td></tr>"
        sh_html += "</tbody></table>"
        st.markdown(sh_html, unsafe_allow_html=True)
        st.markdown("<br>**Accepted column names**", unsafe_allow_html=True)

        WASH_COLS = [
            ("Run",                   True,  "Run, Run No, Run Number, ID"),
            ("Temp (C)",              True,  "Temperature (C), Temp (°C), Temperature (°C), Temp"),
            ("Time (min)",            True,  "Time (minutes), Duration (min), t (min)"),
            ("Water:dust (mL/g)",     True,  "Water:dust (ml/g), Water/dust (mL/g), Water ratio (mL/g), Ratio (mL/g)"),
            ("Initial dust M0 (g)",   True,  "M0 (g), Initial mass (g), Initial weight (g), M0"),
            ("Final dry mass Md (g)", True,  "Md (g), Final mass (g), Dry mass (g), Md"),
        ]
        RUNS_COLS = [
            ("Wash run",              True,  "Wash, Wash No, Washing run, Run (wash)"),
            ("Temp (C)",              True,  "— same as Washing sheet —"),
            ("Time (min)",            True,  "— same as Washing sheet —"),
            ("Water:dust (mL/g)",     True,  "— same as Washing sheet —"),
            ("Carbon (% of dust)",    True,  "Carbon (%), Carbon dose (%), AC (%), C (%), Activated carbon (%)"),
            ("Date",                  True,  "Run date, Test date, Day, Experiment date"),
            ("HCHO baseline / peak / plateau", False, "HCHO_baseline, hcho baseline, HCHO baseline (ppm) …"),
            ("TVOC baseline / peak / plateau", False, "TVOC_baseline, tvoc baseline …"),
            ("CO baseline / peak / plateau",   False, "CO_baseline, co baseline …"),
            ("CO2 baseline / peak / plateau",  False, "CO2_baseline, co2 baseline …"),
        ]

        def _col_rows(rows):
            html = ""
            for canon, required, accepted in rows:
                req_badge = '<span class="req">Required</span>' if required else '<span class="opt">Optional</span>'
                html += (
                    f"<tr><td class='canon'>{canon}</td>"
                    f"<td class='accept'>{accepted}</td>"
                    f"<td>{req_badge}</td></tr>"
                )
            return html

        table_html = """
        <table class="col-map-table">
        <thead><tr><th>Canonical column name</th><th>Also accepted (examples)</th><th></th></tr></thead>
        <tbody>
        <tr><td colspan="3" style="background:var(--sage-soft);font-weight:600;
            padding:8px 12px;font-size:0.82rem;color:var(--brand);">
            Sheet: Washing</td></tr>
        {wash}
        <tr><td colspan="3" style="background:var(--sage-soft);font-weight:600;
            padding:8px 12px;font-size:0.82rem;color:var(--brand);">
            Sheet: PETER_runs</td></tr>
        {runs}
        </tbody></table>
        """.format(wash=_col_rows(WASH_COLS), runs=_col_rows(RUNS_COLS))
        st.markdown(table_html, unsafe_allow_html=True)

        callout("note",
                "If your column is not in the accepted list, rename it to match the "
                "<b>Canonical column name</b> exactly, then re-upload.")
        if show_guide:
            _set("show_col_guide", False)   # reset so it doesn't auto-expand next time


# ══════════════════════════════════════════ PAGE: Q1
elif page_key == "q1":
    page_header(
        "Q1 Washing",
        "Does hot-water washing reduce emissions?",
        "Run Part A to fit the yield model, then Part B to analyse HCHO, TVOC, CO and CO₂. "
        "Work through them in order — Part B uses the yield model from Part A.",
    )

    wb = _get("wb_path")
    if not wb:
        st.warning("Upload your workbook on the **Data** page first."); st.stop()
    wash = _get("wash"); runs = _get("runs")

    # ── Part A
    with panel(
        "parta", "Part A: Yield model (Md)",
        "Fits a <b>second-order response surface</b> for final dry mass after washing. "
        "Reports R², adjusted R², predicted R² (PRESS), Box-Cox transform check, "
        "Type II ANOVA, lack of fit, residual diagnostics and the reduced model equation.",
    ):
        if st.button("Run yield model", type="primary"):
            with st.spinner("Fitting response surface for Md…"):
                wm, out1 = capture(pa.part1, wash, OUT)
            _set("wash_model", wm); _set("part1_log", out1)
            st.success("Yield model fitted.")

        if _get("wash_model"):
            with st.expander("Full model output"):
                st.code(_get("part1_log"), language="")
            c1, c2 = st.columns(2)
            with c1: _img(os.path.join(OUT, "part1_fit.png"),  "Observed vs fitted Md")
            with c2: _img(os.path.join(OUT, "diag_part1.png"), "Residual diagnostics")
            eq_path = os.path.join(OUT, "equation_Md.txt")
            if os.path.exists(eq_path):
                with st.expander("Model equation"):
                    with open(eq_path) as fh: st.code(fh.read())
            d1, d2, _sp = st.columns([1.2, 1.2, 4])
            with d1: _dl(eq_path, "Equation (.txt)")
            with d2: _dl(os.path.join(OUT, "part1_predictions.csv"), "Predictions (.csv)")

    # ── Part B
    with panel(
        "partb", "Part B: Emission analysis (split-plot)",
        "Uses a <b>linear mixed model</b> (wash as random intercept) for each gas. "
        "Temperature, time and water ratio are whole-plot factors; "
        "carbon dose is the sub-plot factor tested within each wash.",
    ):
        if _get("wash_model") is None:
            callout("info", "Run Part A above first — the emission analysis uses the yield model.")
        elif st.button("Run emission analysis", type="primary"):
            with st.spinner("Fitting split-plot models for all four gases…"):
                em_results, out2 = capture(pa.part2, runs, OUT, _get("wash_model"))
            _set("em_results", em_results); _set("part2_log", out2)
            st.success("Emission models fitted.")

        if _get("part2_log"):
            with st.expander("Full emission output"):
                st.code(_get("part2_log"), language="")

    if _get("part2_log"):
        bh_path = os.path.join(OUT, "bh_correction.csv")
        if os.path.exists(bh_path):
            with panel(
                "bh", "Multiple-testing correction (Benjamini-Hochberg FDR)",
                "Controls the false-discovery rate across all gas and response-type model F-tests.",
            ):
                st.dataframe(pd.read_csv(bh_path, index_col=0).round(4))
                _dl(bh_path, "BH table (.csv)")

        diag_imgs = sorted(glob.glob(os.path.join(OUT, "diag_*_net_peak.png")))
        if diag_imgs:
            with panel("diag", "Residual diagnostic plots"):
                cols = st.columns(min(len(diag_imgs), 2))
                for i, f in enumerate(diag_imgs):
                    g = os.path.basename(f).replace("diag_", "").replace("_net_peak.png", "")
                    cols[i % 2].image(PILImage.open(f), caption=f"{g} net peak", use_container_width=True)

        eq_files = sorted(glob.glob(os.path.join(OUT, "equation_*_net_peak.txt")))
        if eq_files:
            with panel("eqs", "Reduced whole-plot equations"):
                for f in eq_files:
                    g = os.path.basename(f).replace("equation_", "").replace("_net_peak.txt", "")
                    with st.expander(f"{g} equation"):
                        with open(f) as fh: st.code(fh.read())

        st.button("Continue to Q2", type="primary", on_click=go, args=("q2",))


# ══════════════════════════════════════════ PAGE: Q2
elif page_key == "q2":
    page_header(
        "Q2 Window",
        "What is the operating window?",
        "Find washing conditions that keep yield above the target and cut all four gas "
        "emissions by the required fraction. Adjust the controls, then calculate.",
    )

    wb = _get("wb_path"); wm = _get("wash_model"); runs = _get("runs")
    if not wb:
        st.warning("Upload your workbook on the **Data** page first."); st.stop()
    if wm is None:
        st.warning("Run the yield model on **Q1** first."); st.stop()

    with panel("targets", "Targets"):
        c1, c2, c3 = st.columns(3)
        dose          = c1.selectbox("Carbon dose (%)", [0, 5, 10, 15], index=2)
        yield_min     = c2.slider("Minimum yield (%)", 50.0, 95.0, 80.0, 1.0)
        reduction_pct = c3.slider(
            "Emission cut target (%)", 10, 90, 50, 5, format="%d%%",
            help="Required reduction vs the no-carbon, least-washed reference.",
        )
        reduction = reduction_pct / 100

        if st.button("Calculate operating window", type="primary"):
            with st.spinner("Searching the grid…"):
                ctrl = pa.load_controls(wb)
                grid, out_w = capture(pa.window, wm, runs, OUT, controls=ctrl,
                                      dose=dose, reduction=reduction, yield_min=yield_min)
            _set("window_grid", grid); _set("window_log", out_w)
            st.success("Window calculated.")

        if _get("window_log"):
            with st.expander("Calculation output"):
                st.code(_get("window_log"), language="")

    if _get("window_grid") is not None:
        grid = _get("window_grid")

        with panel("winplots", "Window plots"):
            c1, c2 = st.columns(2)
            with c1: _img(os.path.join(OUT, "window_by_ratio.png"), "🟢 Robust   🟡 Point only   ⚪ Fails")
            with c2: _img(os.path.join(OUT, "window_contour.png"),  "Desirability D (W = 10 mL/g)")

        gas_imgs = sorted(glob.glob(os.path.join(OUT, "contour_*.png")))
        if gas_imgs:
            with panel("gascont", "Per-gas emission contours (W = 10 mL/g)"):
                cols = st.columns(len(gas_imgs))
                for c, f in zip(cols, gas_imgs):
                    g = os.path.basename(f).replace("contour_", "").replace(".png", "")
                    c.image(PILImage.open(f), caption=g, use_container_width=True)

        with panel(
            "feasible", "Feasible settings",
            "T = temperature (°C) · t = time (min) · W = water:dust (mL/g) · "
            "SRY = predicted yield (%) · D = desirability 0–1 · *_rel = emission / reference.",
        ):
            ok = grid[grid.feasible_robust] if grid.feasible_robust.any() else grid[grid.feasible_point]
            if ok.empty:
                st.warning(
                    "No setting meets every target at these values. "
                    "Lower the emission cut (try 25%) or the yield target and recalculate."
                )
            else:
                rel_cols  = [c for c in ok.columns if c.endswith("_rel") and not c.endswith("_rel_hi")]
                show_cols = ["T", "t", "W", "SRY", "desirability"] + rel_cols
                st.dataframe(ok[show_cols].sort_values("desirability", ascending=False).head(20).round(3))
            d1, d2, _sp = st.columns([1.6, 1.6, 4])
            with d1: _dl(os.path.join(OUT, "operating_window.csv"),      "Feasible settings (.csv)")
            with d2: _dl(os.path.join(OUT, "operating_window_grid.csv"), "Full grid (.csv)")

        st.button("Continue to Q3", type="primary", on_click=go, args=("q3",))


# ══════════════════════════════════════════ PAGE: Q3
elif page_key == "q3":
    page_header(
        "Q3 Carbon",
        "Does activated carbon reduce emissions further?",
        "Compare emission levels across the four carbon doses (0, 5, 10 and 15%) "
        "and test whether the carbon effect is statistically significant.",
    )

    wb = _get("wb_path"); runs = _get("runs")
    if not wb:
        st.warning("Upload your workbook on the **Data** page first."); st.stop()

    with panel("dose", "Dose–response"):
        c1, c2 = st.columns([1, 3])
        gas         = c1.selectbox("Gas", pa.GASES)
        kind_choice = c1.radio("Response", ["net_peak", "net_plateau"])
        y = f"{gas}_{kind_choice}"

        if y in runs.columns and runs[y].notna().sum() > 0:
            with plt.rc_context(MPL_RC):
                fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
                means = runs.groupby("Cpct")[y].mean()
                axes[0].bar(means.index.astype(str), means.values, color=BRAND, width=0.55)
                axes[0].set(xlabel="Carbon dose (%)", ylabel=f"{gas} {kind_choice} (mean)",
                            title="Mean by dose")
                for T, g_df in runs.groupby("T"):
                    m = g_df.groupby("Cpct")[y].mean()
                    axes[1].plot(m.index, m.values, marker="o", markersize=5, linewidth=2,
                                 label=f"{T} °C")
                axes[1].set_xticks(sorted(runs["Cpct"].dropna().unique()))
                axes[1].set(xlabel="Carbon dose (%)", ylabel=f"{gas}", title="Dose × temperature")
                axes[1].legend(fontsize=8)
            with c2:
                st.pyplot(fig)
            plt.close(fig)
        else:
            c2.info(f"No data for {y}.")

    with panel("terms", "Carbon sub-plot model terms"):
        em_results = _get("em_results")
        if em_results is None:
            callout("info", "Run the emission analysis on Q1 Part B first.")
        else:
            for key in [f"{gas}_net_peak", f"{gas}_net_plateau"]:
                if key not in em_results: continue
                mm = em_results[key]
                pv = mm.pvalues.drop([i for i in mm.pvalues.index if i == "Group Var"], errors="ignore")
                carbon_terms = pv[[i for i in pv.index if "Cc" in i or "carbon" in i.lower()]]
                if len(carbon_terms) == 0:
                    st.write(f"No carbon terms in {key} model."); continue
                df_ct = pd.DataFrame({"coef": mm.params[carbon_terms.index],
                                       "p":    carbon_terms}).round(4)
                df_ct["significant"] = df_ct["p"] < 0.05
                st.caption(f"**{key}**")
                st.dataframe(df_ct.style.map(
                    lambda v: "background-color:#d9e7de" if v else "", subset=["significant"]))

    with panel("boxes", "Net peak by dose, all gases"):
        with plt.rc_context(MPL_RC):
            fig2, ax2 = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
            for ax, g in zip(ax2.flat, pa.GASES):
                yg = f"{g}_net_peak"
                if yg not in runs or runs[yg].isna().all():
                    ax.set_visible(False); continue
                doses = sorted(runs["Cpct"].dropna().unique())
                data  = [runs.loc[runs["Cpct"] == d, yg].dropna().values for d in doses]
                bp = ax.boxplot(
                    data, patch_artist=True, widths=0.5,
                    medianprops=dict(color=INK, linewidth=1.6),
                    flierprops=dict(marker="o", markersize=3,
                                    markerfacecolor="#8c978f", markeredgecolor="none"),
                )
                for patch in bp["boxes"]:
                    patch.set(facecolor=SAGE, edgecolor=BRAND, linewidth=1.2)
                for line in bp["whiskers"] + bp["caps"]:
                    line.set(color=BRAND, linewidth=1.1)
                ax.set_xticks(range(1, len(doses) + 1), [f"{d:g}" for d in doses])
                ax.set(title=g, xlabel="Carbon dose (%)", ylabel="Net peak")
            fig2.suptitle("Net peak by carbon dose", fontweight="600", fontsize=12)
        st.pyplot(fig2)
        plt.close(fig2)

    bh_path = os.path.join(OUT, "bh_correction.csv")
    if os.path.exists(bh_path):
        with panel("bh3", "Multiple-testing correction",
                   "BH-adjusted p-values control the false-discovery rate."):
            st.dataframe(pd.read_csv(bh_path, index_col=0).round(4))

    with panel("guide", "Interpretation guide"):
        st.markdown(
            "- **Cc**: linear dose effect. Significant means more carbon changes emissions proportionally.  \n"
            "- **I(Cc²)**: curved effect. Significant means diminishing returns or an optimum dose.  \n"
            "- **Cc:x1 / Cc:x2 / Cc:x3**: the carbon effect depends on temperature, time or ratio.  \n"
            "- PM2.5 and PM10 are excluded because the sensor was at ceiling throughout all runs."
        )
