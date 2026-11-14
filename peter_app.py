"""PETER Study — Streamlit Web Interface
Run:  streamlit run peter_app.py
"""

import os, sys, pathlib, importlib.util, re, io, contextlib, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image as PILImage

st.set_page_config(
    page_title="PETER Study",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────── CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #f5f6f8;
    color: #111827;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Top navigation bar ── */
.topnav {
    position: sticky;
    top: 0;
    z-index: 999;
    background: #ffffff;
    border-bottom: 1px solid #e5e7eb;
    display: flex;
    align-items: center;
    padding: 0 32px;
    height: 60px;
    gap: 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.topnav-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-right: 40px;
    text-decoration: none;
}
.topnav-brand-icon {
    width: 32px;
    height: 32px;
    background: #0f4c35;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    color: white;
    flex-shrink: 0;
}
.topnav-brand-text {
    font-size: 0.95rem;
    font-weight: 700;
    color: #0f4c35;
    letter-spacing: -0.01em;
    white-space: nowrap;
}
.topnav-brand-sub {
    font-size: 0.7rem;
    color: #6b7280;
    font-weight: 400;
    white-space: nowrap;
}
.nav-links {
    display: flex;
    align-items: center;
    gap: 2px;
    flex: 1;
}
.nav-link {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 0.84rem;
    font-weight: 500;
    color: #374151;
    text-decoration: none;
    white-space: nowrap;
    cursor: pointer;
    border: none;
    background: none;
    transition: background 0.15s, color 0.15s;
}
.nav-link:hover { background: #f3f4f6; color: #111827; }
.nav-link.active {
    background: #ecfdf5;
    color: #0f4c35;
    font-weight: 600;
}
.nav-link .step-dot {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: #e5e7eb;
    color: #6b7280;
    font-size: 0.65rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.nav-link.active .step-dot {
    background: #0f4c35;
    color: #fff;
}
.nav-link.done .step-dot {
    background: #10b981;
    color: #fff;
}
.nav-link.locked { opacity: 0.4; cursor: not-allowed; }
.nav-sep {
    width: 1px;
    height: 20px;
    background: #e5e7eb;
    margin: 0 6px;
}
.nav-badge {
    font-size: 0.65rem;
    background: #dcfce7;
    color: #15803d;
    border-radius: 4px;
    padding: 1px 6px;
    font-weight: 600;
}

/* ── Page shell ── */
.page-shell {
    max-width: 1200px;
    margin: 0 auto;
    padding: 36px 32px 60px;
}

/* ── Page header ── */
.page-header { margin-bottom: 28px; }
.page-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    color: #111827;
    letter-spacing: -0.02em;
    line-height: 1.25;
}
.page-header p {
    font-size: 0.9rem;
    color: #6b7280;
    margin-top: 6px;
    line-height: 1.6;
}
.breadcrumb {
    font-size: 0.75rem;
    color: #9ca3af;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Cards ── */
.card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
}
.card-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #111827;
    margin-bottom: 4px;
}
.card-desc {
    font-size: 0.82rem;
    color: #6b7280;
    margin-bottom: 16px;
    line-height: 1.5;
}

/* ── Stat row ── */
.stat-row {
    display: flex;
    gap: 12px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}
.stat-box {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 16px 20px;
    flex: 1;
    min-width: 110px;
}
.stat-box .s-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #9ca3af;
    margin-bottom: 6px;
}
.stat-box .s-val {
    font-size: 1.6rem;
    font-weight: 700;
    color: #0f4c35;
    line-height: 1;
}
.stat-box .s-sub {
    font-size: 0.72rem;
    color: #9ca3af;
    margin-top: 2px;
}

/* ── Section heading inside page ── */
.section-heading {
    font-size: 1rem;
    font-weight: 600;
    color: #111827;
    padding-bottom: 10px;
    border-bottom: 2px solid #e5e7eb;
    margin: 28px 0 16px;
}

/* ── Step callout ── */
.step-callout {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 16px;
}
.step-callout .sc-num {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #0f4c35;
    color: #fff;
    font-size: 0.75rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
}
.step-callout .sc-body .sc-title {
    font-size: 0.88rem;
    font-weight: 600;
    color: #111827;
    margin-bottom: 2px;
}
.step-callout .sc-body .sc-text {
    font-size: 0.82rem;
    color: #6b7280;
    line-height: 1.5;
}

/* ── Info/note box ── */
.note-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: #78350f;
    line-height: 1.6;
}
.info-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: #1e40af;
    line-height: 1.6;
}

/* ── Streamlit overrides ── */
div[data-testid="stButton"] > button {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.84rem;
    border-radius: 8px;
    padding: 9px 20px;
    transition: all 0.15s;
}
div[data-testid="stButton"] > button[kind="primary"] {
    background: #0f4c35;
    border: none;
    color: #fff;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #0d3d2a;
    box-shadow: 0 2px 8px rgba(15,76,53,0.25);
}
div[data-testid="stButton"] > button[kind="secondary"] {
    background: #fff;
    border: 1px solid #d1d5db;
    color: #374151;
}
[data-testid="stExpander"] {
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    background: #fff !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.85rem;
    font-weight: 500;
}
[data-testid="stFileUploader"] {
    border-radius: 10px;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stRadio"] label {
    font-size: 0.82rem;
    font-weight: 500;
    color: #374151;
}
[data-testid="stDataFrame"] {
    border-radius: 8px;
    border: 1px solid #e5e7eb;
    overflow: hidden;
}
[data-testid="stAlert"] { border-radius: 8px; font-size: 0.84rem; }
p, li { font-size: 0.88rem; line-height: 1.65; color: #374151; }
h3 { font-size: 1rem; font-weight: 600; color: #111827; }
</style>
""", unsafe_allow_html=True)


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

def stat(label, value, sub=""):
    return (f'<div class="stat-box"><div class="s-label">{label}</div>'
            f'<div class="s-val">{value}</div>'
            f'<div class="s-sub">{sub}</div></div>')

def step_note(num, title, text):
    st.markdown(
        f'<div class="step-callout">'
        f'<div class="sc-num">{num}</div>'
        f'<div class="sc-body"><div class="sc-title">{title}</div>'
        f'<div class="sc-text">{text}</div></div></div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────── top nav
data_ok = _get("wb_path") is not None
q1a_ok  = _get("wash_model") is not None
q1b_ok  = _get("em_results") is not None

PAGES = [
    ("load", "Data",           "Load & inspect",    data_ok),
    ("q1",   "Q1 — Washing",   "Yield & emissions", q1a_ok),
    ("q2",   "Q2 — Window",    "Operating window",  _get("window_grid") is not None),
    ("q3",   "Q3 — Carbon",    "Carbon effect",     False),
]

# page selection via query params (fallback to session)
qp = st.query_params
if "page" not in qp:
    qp["page"] = _get("current_page", "load")
page_key = qp.get("page", "load")
_set("current_page", page_key)

def nav_item(key, num, short, long, done):
    is_active = page_key == key
    cls = "nav-link active" if is_active else ("nav-link done" if done else "nav-link")
    dot = "✓" if done else str(num)
    dot_cls = "step-dot"
    badge = f'<span class="nav-badge">Done</span>' if done and not is_active else ""
    return (f'<a class="{cls}" href="?page={key}">'
            f'<span class="{dot_cls}">{dot}</span>'
            f'<span>{short}</span>{badge}</a>')

nav_html = (
    '<nav class="topnav">'
    '<div class="topnav-brand">'
    '<div class="topnav-brand-icon">🌲</div>'
    '<div><div class="topnav-brand-text">PETER Study</div>'
    '<div class="topnav-brand-sub">Analysis Pipeline</div></div>'
    '</div>'
    '<div class="nav-sep"></div>'
    '<div class="nav-links">'
    + nav_item("load", 1, "Data",         "Load & inspect",    data_ok)
    + nav_item("q1",   2, "Q1 — Washing", "Yield & emissions", q1a_ok)
    + nav_item("q2",   3, "Q2 — Window",  "Operating window",  _get("window_grid") is not None)
    + nav_item("q3",   4, "Q3 — Carbon",  "Carbon effect",     False)
    + '</div>'
    + f'<div style="margin-left:auto;font-size:0.72rem;color:#9ca3af;">v{pa_ver}</div>'
    + '</nav>'
)
st.markdown(nav_html, unsafe_allow_html=True)


# ══════════════════════════════════════════ helpers for page shell
def page_open(crumb, title, subtitle):
    st.markdown('<div class="page-shell">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="page-header">'
        f'<div class="breadcrumb">PETER Study <span>›</span> {crumb}</div>'
        f'<h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )

def page_close():
    st.markdown('</div>', unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-heading">{title}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════ PAGE: LOAD DATA
if page_key == "load":
    page_open(
        "Load Data",
        "Load your workbook",
        "Upload PETER_Data_Workbook.xlsx to begin. All analysis pages use this file. "
        "Use the simulated workbook to test the pipeline before the real data is ready.",
    )

    uploaded = st.file_uploader("Choose your workbook (.xlsx)", type=["xlsx", "xlsm"])
    if uploaded:
        wb_path = os.path.join(OUT, uploaded.name)
        with open(wb_path, "wb") as f: f.write(uploaded.read())
        try:
            pa.check_workbook(wb_path)
            _set("wb_path", wb_path)
            wash = pa.load_washing(wb_path)
            runs = pa.load_runs(wb_path)
            _set("wash", wash); _set("runs", runs)
            st.success(f"✅ Loaded: **{uploaded.name}**")
        except Exception as e:
            st.error(f"Problem reading workbook: {e}"); st.stop()

    wb = _get("wb_path")
    if wb:
        wash = _get("wash"); runs = _get("runs")

        # stat row
        st.markdown(
            '<div class="stat-row">'
            + stat("Wash runs",    len(wash))
            + stat("Md recorded",  wash["Md"].notna().sum(), "final dry mass")
            + stat("PETER rows",   len(runs))
            + stat("Washes",       runs["wash"].nunique())
            + stat("Carbon doses", runs["Cpct"].nunique())
            + '</div>',
            unsafe_allow_html=True,
        )

        section("Sheet previews")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("**Washing sheet**")
            st.dataframe(wash[["Run", "T", "t", "W", "Md", "SRY"]].round(2), height=260)
        with c2:
            st.caption("**PETER_runs sheet** (first 20 rows)")
            st.dataframe(runs[["wash", "T", "t", "W", "Cpct",
                                "HCHO_net_peak", "TVOC_net_peak",
                                "CO_net_peak", "CO2_net_peak"]].head(20).round(2), height=260)

        section("Missing values")
        cols = ["Md"] + [c for c in runs.columns if c.endswith(("net_peak", "net_plateau"))]
        mv   = pd.concat([
            wash[["Md"]].isna().sum().rename("Washing"),
            runs[[c for c in cols[1:] if c in runs]].isna().sum().rename("PETER_runs"),
        ], axis=1).fillna("")
        st.dataframe(mv)

        st.markdown(
            '<div class="note-box">ℹ️ Empty cells in Md or any emission column are fine — '
            'the analysis skips any response with fewer than 20 non-missing values.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<br><a href="?page=q1" style="display:inline-block;background:#0f4c35;'
            'color:#fff;padding:10px 24px;border-radius:8px;font-size:0.85rem;'
            'font-weight:600;text-decoration:none;">Continue to Q1 →</a>',
            unsafe_allow_html=True,
        )
    else:
        step_note(1, "Upload your workbook", "Click the file picker above to choose your .xlsx file.")
        step_note(2, "Inspect both sheets", "A preview of the Washing and PETER_runs sheets appears here.")
        step_note(3, "Navigate to Q1", "Use the top navigation bar to move through the analysis in order.")

    page_close()


# ══════════════════════════════════════════ PAGE: Q1
elif page_key == "q1":
    page_open(
        "Q1 — Washing",
        "Does hot-water washing reduce emissions?",
        "Run Part A to fit the yield model, then Part B to analyse HCHO, TVOC, CO and CO₂. "
        "Work through them in order — Part B uses the yield model from Part A.",
    )

    wb = _get("wb_path")
    if not wb:
        st.warning("Upload your workbook on the **Data** page first."); page_close(); st.stop()
    wash = _get("wash"); runs = _get("runs")

    # ── Part A
    section("Part A — Yield model (Md)")
    st.markdown(
        "Fits a **second-order response surface** for final dry mass after washing. "
        "Reports R², adjusted R², predicted R² (PRESS), Box-Cox transform check, "
        "Type II ANOVA, lack of fit, residual diagnostics and the reduced model equation."
    )

    if st.button("Run yield model", type="primary"):
        with st.spinner("Fitting response surface for Md…"):
            wm, out1 = capture(pa.part1, wash, OUT)
        _set("wash_model", wm); _set("part1_log", out1)
        st.success("Yield model fitted.")

    if _get("wash_model"):
        with st.expander("Full model output"):
            st.code(_get("part1_log"), language="")
        c1, c2 = st.columns(2)
        with c1: _img(os.path.join(OUT, "part1_fit.png"), "Observed vs fitted Md")
        with c2: _img(os.path.join(OUT, "diag_part1.png"), "Residual diagnostics")
        eq_path = os.path.join(OUT, "equation_Md.txt")
        if os.path.exists(eq_path):
            with st.expander("Model equation"):
                st.code(open(eq_path).read())
            _dl(eq_path, "Download equation (.txt)")
        _dl(os.path.join(OUT, "part1_predictions.csv"), "Download predictions (.csv)")

    # ── Part B
    section("Part B — Emission analysis (split-plot)")
    st.markdown(
        "Uses a **linear mixed model** (wash = random intercept) for each gas. "
        "Temperature, time and water ratio are whole-plot factors; "
        "carbon dose is the sub-plot factor tested within each wash."
    )

    if _get("wash_model") is None:
        st.markdown(
            '<div class="info-box">Run Part A above first — the emission analysis '
            'uses the yield model.</div>', unsafe_allow_html=True,
        )
    elif st.button("Run emission analysis", type="primary"):
        with st.spinner("Fitting split-plot models for all four gases…"):
            em_results, out2 = capture(pa.part2, runs, OUT, _get("wash_model"))
        _set("em_results", em_results); _set("part2_log", out2)
        st.success("Emission models fitted.")

    if _get("part2_log"):
        with st.expander("Full emission output"):
            st.code(_get("part2_log"), language="")

        bh_path = os.path.join(OUT, "bh_correction.csv")
        if os.path.exists(bh_path):
            section("Multiple-testing correction (Benjamini-Hochberg FDR)")
            st.caption("Controls the false-discovery rate across all gas × response-type model F-tests.")
            st.dataframe(pd.read_csv(bh_path, index_col=0).round(4))
            _dl(bh_path, "Download BH table (.csv)")

        section("Residual diagnostic plots")
        diag_imgs = sorted(glob.glob(os.path.join(OUT, "diag_*_net_peak.png")))
        if diag_imgs:
            cols = st.columns(min(len(diag_imgs), 2))
            for i, f in enumerate(diag_imgs):
                g = os.path.basename(f).replace("diag_", "").replace("_net_peak.png", "")
                cols[i % 2].image(PILImage.open(f), caption=f"{g} net peak", use_container_width=True)

        section("Reduced whole-plot equations")
        for f in sorted(glob.glob(os.path.join(OUT, "equation_*_net_peak.txt"))):
            g = os.path.basename(f).replace("equation_", "").replace("_net_peak.txt", "")
            with st.expander(f"{g} equation"):
                st.code(open(f).read())

        st.markdown(
            '<br><a href="?page=q2" style="display:inline-block;background:#0f4c35;'
            'color:#fff;padding:10px 24px;border-radius:8px;font-size:0.85rem;'
            'font-weight:600;text-decoration:none;">Continue to Q2 →</a>',
            unsafe_allow_html=True,
        )

    page_close()


# ══════════════════════════════════════════ PAGE: Q2
elif page_key == "q2":
    page_open(
        "Q2 — Operating Window",
        "What is the operating window?",
        "Find washing conditions that simultaneously keep yield above the target "
        "and cut all four gas emissions by the required fraction. Adjust the controls and calculate.",
    )

    wb = _get("wb_path"); wm = _get("wash_model"); runs = _get("runs")
    if not wb:
        st.warning("Upload your workbook on the **Data** page first."); page_close(); st.stop()
    if wm is None:
        st.warning("Run the yield model on **Q1** first."); page_close(); st.stop()

    section("Targets")
    c1, c2, c3 = st.columns(3)
    dose      = c1.selectbox("Carbon dose (%)", [0, 5, 10, 15], index=2)
    yield_min = c2.slider("Minimum yield (%)", 50.0, 95.0, 80.0, 1.0)
    reduction = c3.slider("Emission cut target", 0.10, 0.90, 0.50, 0.05, format="%.0f%%",
                           help="Required reduction vs the no-carbon, least-washed reference.")

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

        section("Window plots")
        c1, c2 = st.columns(2)
        with c1: _img(os.path.join(OUT, "window_by_ratio.png"), "🟢 Robust   🟡 Point only   ⚪ Fails")
        with c2: _img(os.path.join(OUT, "window_contour.png"),  "Desirability D (W = 10 mL/g)")

        section("Per-gas emission contours (W = 10 mL/g)")
        gas_imgs = sorted(glob.glob(os.path.join(OUT, "contour_*.png")))
        if gas_imgs:
            cols = st.columns(len(gas_imgs))
            for c, f in zip(cols, gas_imgs):
                g = os.path.basename(f).replace("contour_", "").replace(".png", "")
                c.image(PILImage.open(f), caption=g, use_container_width=True)

        section("Feasible settings")
        ok = grid[grid.feasible_robust] if grid.feasible_robust.any() else grid[grid.feasible_point]
        if ok.empty:
            st.warning(
                "No setting meets every target at these values. "
                "Lower the emission cut (try 25%) or yield target and recalculate."
            )
        else:
            rel_cols  = [c for c in ok.columns if c.endswith("_rel") and not c.endswith("_rel_hi")]
            show_cols = ["T", "t", "W", "SRY", "desirability"] + rel_cols
            st.dataframe(ok[show_cols].sort_values("desirability", ascending=False).head(20).round(3))
            st.caption(
                "T = temperature (°C) · t = time (min) · W = water:dust (mL/g) · "
                "SRY = predicted yield (%) · D = desirability 0–1 · *_rel = emission / reference"
            )
        _dl(os.path.join(OUT, "operating_window.csv"),      "Download feasible settings (.csv)")
        _dl(os.path.join(OUT, "operating_window_grid.csv"), "Download full grid (.csv)")

        st.markdown(
            '<br><a href="?page=q3" style="display:inline-block;background:#0f4c35;'
            'color:#fff;padding:10px 24px;border-radius:8px;font-size:0.85rem;'
            'font-weight:600;text-decoration:none;">Continue to Q3 →</a>',
            unsafe_allow_html=True,
        )

    page_close()


# ══════════════════════════════════════════ PAGE: Q3
elif page_key == "q3":
    page_open(
        "Q3 — Carbon Effect",
        "Does activated carbon reduce emissions further?",
        "Compare emission levels across the four carbon doses (0, 5, 10, 15%) "
        "and test whether the carbon effect is statistically significant.",
    )

    wb = _get("wb_path"); runs = _get("runs")
    if not wb:
        st.warning("Upload your workbook on the **Data** page first."); page_close(); st.stop()

    section("Dose–response")
    c1, c2 = st.columns([1, 3])
    gas         = c1.selectbox("Gas", pa.GASES)
    kind_choice = c1.radio("Response", ["net_peak", "net_plateau"])
    y = f"{gas}_{kind_choice}"

    if y in runs.columns and runs[y].notna().sum() > 0:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        means = runs.groupby("Cpct")[y].mean()
        axes[0].bar(means.index.astype(str), means.values, color="#0f4c35",
                    width=0.5, edgecolor="white", linewidth=0.8)
        axes[0].set(xlabel="Carbon dose (%)", ylabel=f"{gas} {kind_choice} (mean)",
                    title="Mean by dose")
        for T, g_df in runs.groupby("T"):
            m = g_df.groupby("Cpct")[y].mean()
            axes[1].plot(m.index, m.values, marker="o", linewidth=1.8, label=f"{T} °C")
        axes[1].set(xlabel="Carbon dose (%)", ylabel=f"{gas}", title="Dose × temperature")
        axes[1].legend(fontsize=8)
        plt.tight_layout()
        with c2: st.pyplot(fig); plt.close(fig)
    else:
        st.info(f"No data for {y}.")

    section("Carbon sub-plot model terms")
    em_results = _get("em_results")
    if em_results is None:
        st.markdown(
            '<div class="info-box">Run the emission analysis on Q1 Part B first.</div>',
            unsafe_allow_html=True,
        )
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
                lambda v: "background-color:#dcfce7" if v else "", subset=["significant"]))

    section("Net peak by dose — all gases")
    fig2, ax2 = plt.subplots(2, 2, figsize=(12, 7))
    for ax, g in zip(ax2.flat, pa.GASES):
        yg = f"{g}_net_peak"
        if yg not in runs or runs[yg].isna().all():
            ax.set_visible(False); continue
        runs.boxplot(column=yg, by="Cpct", ax=ax)
        ax.set(title=g, xlabel="Carbon dose (%)", ylabel="Net peak")
        plt.sca(ax); plt.title(g)
    plt.suptitle("Net peak by carbon dose", y=1.01)
    plt.tight_layout()
    st.pyplot(fig2); plt.close(fig2)

    bh_path = os.path.join(OUT, "bh_correction.csv")
    if os.path.exists(bh_path):
        section("Multiple-testing correction")
        st.dataframe(pd.read_csv(bh_path, index_col=0).round(4))
        st.caption("BH-adjusted p-values control the false-discovery rate.")

    section("Interpretation guide")
    st.markdown(
        "- **Cc** — linear dose effect: significant means more carbon changes emissions proportionally  \n"
        "- **I(Cc²)** — curved effect: significant means diminishing returns or an optimum dose  \n"
        "- **Cc:x1 / Cc:x2 / Cc:x3** — the carbon effect depends on temperature / time / ratio  \n"
        "- PM2.5 and PM10 are excluded (sensor at ceiling throughout all runs)  \n"
        "- Results from the **simulated workbook** only test the code and are not project findings"
    )

    page_close()
