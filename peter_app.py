"""PETER Study — Streamlit Web Interface
Run:  streamlit run peter_app.py
"""

import os, sys, pathlib, importlib.util, re, io, contextlib, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image as PILImage

# ─────────────────────────────────────────────── page config (must be first)
st.set_page_config(
    page_title="PETER Study",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────── global CSS
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Colour tokens ── */
:root {
    --green-dark:  #1a6b55;
    --green-mid:   #2a9d8f;
    --green-light: #52b788;
    --amber:       #f4a261;
    --bg-card:     #f8fffe;
    --text-main:   #1a1a2e;
    --text-muted:  #5a6a72;
    --border:      #d4e9e2;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d3b2e 0%, #1a6b55 100%);
    border-right: 1px solid #0a2a1f;
}
[data-testid="stSidebar"] * {
    color: #e8f5f0 !important;
}
[data-testid="stSidebar"] .step-item {
    color: #e8f5f0 !important;
}

/* ── Step nav items ── */
.step-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 8px;
    margin-bottom: 6px;
    cursor: pointer;
    transition: background 0.2s;
    font-size: 0.88rem;
    font-weight: 500;
}
.step-item.active {
    background: rgba(255,255,255,0.18);
    border-left: 3px solid #52b788;
}
.step-item.done {
    background: rgba(82,183,136,0.15);
}
.step-item.locked {
    opacity: 0.45;
}
.step-num {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
}
.step-num.active  { background: #52b788; color: #0d3b2e; }
.step-num.done    { background: #2a9d8f; color: #fff; }
.step-num.locked  { background: rgba(255,255,255,0.15); color: #e8f5f0; }

/* ── Page title ── */
.page-title {
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--green-dark);
    margin-bottom: 4px;
}
.page-subtitle {
    font-size: 0.95rem;
    color: var(--text-muted);
    margin-bottom: 24px;
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 20px;
    flex: 1;
    min-width: 140px;
}
.metric-card .label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}
.metric-card .value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--green-dark);
}

/* ── Section divider ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(232,245,240,0.6);
    padding: 10px 14px 4px;
    margin-top: 6px;
}

/* ── Buttons ── */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #2a9d8f, #1a6b55);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
    font-family: 'Inter', sans-serif;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #52b788, #2a9d8f);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(42,157,143,0.3);
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    background: var(--bg-card) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
}

/* ── Info / warning banners ── */
[data-testid="stAlert"] {
    border-radius: 8px;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────── load pa module
@st.cache_resource
def load_pa():
    here   = pathlib.Path(__file__).parent
    script = here / "peter_analysis.py"
    if not script.exists():
        st.error(f"peter_analysis.py not found in {here}.")
        st.stop()
    text = script.read_text(encoding="utf-8", errors="ignore")
    m    = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    ver  = m.group(1) if m else "unknown"
    sys.modules.pop("peter_analysis", None)
    spec = importlib.util.spec_from_file_location("peter_analysis", script)
    pa   = importlib.util.module_from_spec(spec)
    sys.modules["peter_analysis"] = pa
    spec.loader.exec_module(pa)
    import matplotlib
    matplotlib.use("Agg")
    return pa, ver

pa, pa_ver = load_pa()

# ─────────────────────────────────────────────── session helpers
OUT = "results"
os.makedirs(OUT, exist_ok=True)

def _get(key, default=None):  return st.session_state.get(key, default)
def _set(key, val):           st.session_state[key] = val

def _img(path, caption=""):
    if os.path.exists(path):
        st.image(PILImage.open(path), caption=caption, use_container_width=True)

def _dl(path, label="Download"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            st.download_button(label, f.read(), file_name=os.path.basename(path))

def capture(func, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = func(*args, **kwargs)
    return result, buf.getvalue()

def metric_card(label, value):
    return (f'<div class="metric-card">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            f'</div>')

# ─────────────────────────────────────────────── sidebar
# Determine completed steps
data_ok   = _get("wb_path") is not None
q1a_ok    = _get("wash_model") is not None
q1b_ok    = _get("em_results") is not None

def _step(num, label, state):
    cls_item = f"step-item {state}"
    cls_num  = f"step-num {state}"
    icon     = "✓" if state == "done" else ("▶" if state == "active" else str(num))
    return (f'<div class="{cls_item}">'
            f'<div class="{cls_num}">{icon}</div>'
            f'<span>{label}</span>'
            f'</div>')

with st.sidebar:
    # Logo + title
    st.markdown(
        '<div style="text-align:center;padding:20px 0 10px;">'
        '<div style="font-size:2.2rem">🌲</div>'
        '<div style="font-size:1.1rem;font-weight:700;letter-spacing:0.03em;margin-top:4px">PETER Study</div>'
        '<div style="font-size:0.72rem;opacity:0.6;margin-top:2px">Analysis Pipeline</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="border-color:rgba(255,255,255,0.15);margin:0 0 10px"/>', unsafe_allow_html=True)

    # Step progress nav
    st.markdown('<div class="section-label">Workflow</div>', unsafe_allow_html=True)

    steps = [
        ("Load Data",          "done" if data_ok else "active"),
        ("Q1 — Yield model",   "done" if q1a_ok else ("active" if data_ok else "locked")),
        ("Q1 — Emissions",     "done" if q1b_ok else ("active" if q1a_ok else "locked")),
        ("Q2 — Window",        "active" if q1b_ok else "locked"),
        ("Q3 — Carbon effect", "active" if q1b_ok else "locked"),
    ]
    step_html = "".join(_step(i + 1, lbl, st_) for i, (lbl, st_) in enumerate(steps))
    st.markdown(step_html, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(255,255,255,0.15);margin:10px 0"/>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Navigate</div>', unsafe_allow_html=True)

    pages = {
        "🗂  Load Data":           "load",
        "Q1  Does washing work?":  "q1",
        "Q2  Operating window":    "q2",
        "Q3  Carbon effect":       "q3",
    }
    page     = st.radio("", list(pages.keys()), label_visibility="collapsed")
    page_key = pages[page]

    st.markdown('<hr style="border-color:rgba(255,255,255,0.15);margin:10px 0"/>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:0.72rem;opacity:0.5;text-align:center;padding-bottom:12px">'
        f'Pipeline v{pa_ver} · UI by Dr. Olu</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════ PAGE: LOAD DATA
if page_key == "load":
    st.markdown('<div class="page-title">🗂 Load your workbook</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Upload the PETER_Data_Workbook.xlsx to begin. '
        'All analysis pages use this file.</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Choose your workbook (.xlsx)", type=["xlsx", "xlsm"])
    if uploaded:
        wb_path = os.path.join(OUT, uploaded.name)
        with open(wb_path, "wb") as f:
            f.write(uploaded.read())
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

        # Metric summary row
        cards = (
            metric_card("Wash runs", len(wash)) +
            metric_card("Md recorded", wash["Md"].notna().sum()) +
            metric_card("PETER rows", len(runs)) +
            metric_card("Washes", runs["wash"].nunique()) +
            metric_card("Carbon doses", runs["Cpct"].nunique())
        )
        st.markdown(f'<div class="metric-row">{cards}</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Washing sheet**")
            st.dataframe(wash[["Run", "T", "t", "W", "Md", "SRY"]].round(2), height=260)
        with col2:
            st.markdown("**PETER_runs sheet**")
            st.dataframe(runs[["wash", "T", "t", "W", "Cpct",
                                "HCHO_net_peak", "TVOC_net_peak",
                                "CO_net_peak", "CO2_net_peak"]].head(20).round(2), height=260)

        st.markdown("**Missing values in model columns**")
        cols = ["Md"] + [c for c in runs.columns if c.endswith(("net_peak", "net_plateau"))]
        mv   = pd.concat([
            wash[["Md"]].isna().sum().rename("Washing"),
            runs[[c for c in cols[1:] if c in runs]].isna().sum().rename("PETER_runs"),
        ], axis=1).fillna("")
        st.dataframe(mv)
    else:
        st.info("Upload a workbook above to begin. Use the simulated workbook to test the pipeline.")


# ══════════════════════════════════════════ PAGE: Q1
elif page_key == "q1":
    st.markdown('<div class="page-title">Q1 — Does hot-water washing reduce emissions?</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Fits a response-surface model for yield, then a split-plot '
        'emission model for HCHO, TVOC, CO and CO₂. Run Part A first, then Part B.</div>',
        unsafe_allow_html=True,
    )

    wb = _get("wb_path")
    if not wb:
        st.warning("Upload your workbook on **Load Data** first."); st.stop()
    wash = _get("wash"); runs = _get("runs")

    # ── Part A: yield
    st.markdown("### Part A — Yield model (Md)")
    st.markdown(
        "Fits a **second-order response surface** for final dry mass after washing. "
        "Reports R², adjusted R², predicted R² (PRESS), Box-Cox check, ANOVA, lack of fit, "
        "residual diagnostics and the reduced model equation."
    )
    if st.button("▶  Run yield model", type="primary"):
        with st.spinner("Fitting response surface for Md…"):
            wm, out1 = capture(pa.part1, wash, OUT)
        _set("wash_model", wm); _set("part1_log", out1)
        st.success("Yield model fitted.")

    if _get("wash_model"):
        with st.expander("📋 Full model output", expanded=False):
            st.code(_get("part1_log"), language="")
        col1, col2 = st.columns(2)
        with col1: _img(os.path.join(OUT, "part1_fit.png"), "Observed vs fitted Md")
        with col2: _img(os.path.join(OUT, "diag_part1.png"), "Residual diagnostics")
        eq_path = os.path.join(OUT, "equation_Md.txt")
        if os.path.exists(eq_path):
            with st.expander("📐 Model equation"):
                st.code(open(eq_path).read())
            _dl(eq_path, "⬇  Download equation (.txt)")
        _dl(os.path.join(OUT, "part1_predictions.csv"), "⬇  Download predictions (.csv)")

    st.markdown("---")

    # ── Part B: emissions
    st.markdown("### Part B — Emission analysis (split-plot)")
    st.markdown(
        "Uses a **linear mixed model** (wash = random intercept) for each gas. "
        "Temperature, time and water ratio are whole-plot factors; "
        "carbon dose is the sub-plot factor tested within each wash."
    )
    if _get("wash_model") is None:
        st.info("Run Part A above first.")
    elif st.button("▶  Run emission analysis", type="primary"):
        with st.spinner("Fitting split-plot models for all four gases…"):
            em_results, out2 = capture(pa.part2, runs, OUT, _get("wash_model"))
        _set("em_results", em_results); _set("part2_log", out2)
        st.success("Emission models fitted.")

    if _get("part2_log"):
        with st.expander("📋 Full emission output", expanded=False):
            st.code(_get("part2_log"), language="")

        bh_path = os.path.join(OUT, "bh_correction.csv")
        if os.path.exists(bh_path):
            st.markdown("#### Multiple-testing correction (Benjamini-Hochberg FDR)")
            st.caption("Controls the false-discovery rate across all gas × response-type model F-tests.")
            st.dataframe(pd.read_csv(bh_path, index_col=0).round(4))
            _dl(bh_path, "⬇  Download BH table (.csv)")

        st.markdown("#### Residual diagnostic plots")
        diag_imgs = sorted(glob.glob(os.path.join(OUT, "diag_*_net_peak.png")))
        if diag_imgs:
            cols = st.columns(min(len(diag_imgs), 2))
            for i, f in enumerate(diag_imgs):
                gas = os.path.basename(f).replace("diag_", "").replace("_net_peak.png", "")
                cols[i % 2].image(PILImage.open(f), caption=f"{gas} net peak", use_container_width=True)

        st.markdown("#### Reduced whole-plot equations")
        for f in sorted(glob.glob(os.path.join(OUT, "equation_*_net_peak.txt"))):
            gas = os.path.basename(f).replace("equation_", "").replace("_net_peak.txt", "")
            with st.expander(f"📐 {gas}"):
                st.code(open(f).read())


# ══════════════════════════════════════════ PAGE: Q2
elif page_key == "q2":
    st.markdown('<div class="page-title">Q2 — What is the operating window?</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Find washing conditions that keep yield high AND cut all '
        'four gas emissions. Adjust the targets and press Calculate.</div>',
        unsafe_allow_html=True,
    )

    wb = _get("wb_path"); wm = _get("wash_model"); runs = _get("runs")
    if not wb:
        st.warning("Upload your workbook on **Load Data** first."); st.stop()
    if wm is None:
        st.warning("Run the yield model on **Q1 Part A** first."); st.stop()

    col1, col2, col3 = st.columns(3)
    dose      = col1.selectbox("Carbon dose (%)", [0, 5, 10, 15], index=2)
    yield_min = col2.slider("Minimum yield (%)", 50.0, 95.0, 80.0, 1.0)
    reduction = col3.slider("Emission cut target", 0.10, 0.90, 0.50, 0.05, format="%.0f%%",
                             help="Required reduction vs the reference (no-carbon, least-washed corner).")

    if st.button("▶  Calculate operating window", type="primary"):
        with st.spinner("Searching the grid…"):
            ctrl = pa.load_controls(wb)
            grid, out_w = capture(pa.window, wm, runs, OUT, controls=ctrl,
                                  dose=dose, reduction=reduction, yield_min=yield_min)
        _set("window_grid", grid); _set("window_log", out_w)
        st.success("Window calculated.")

    if _get("window_log"):
        with st.expander("📋 Calculation output", expanded=False):
            st.code(_get("window_log"), language="")

    if _get("window_grid") is not None:
        grid = _get("window_grid")

        col1, col2 = st.columns(2)
        with col1:
            _img(os.path.join(OUT, "window_by_ratio.png"),
                 "🟢 Robust   🟡 Point only   ⚫ Fails")
        with col2:
            _img(os.path.join(OUT, "window_contour.png"),
                 "Desirability D at W = 10 mL/g (higher = better)")

        st.markdown("#### Per-gas emission contours (W = 10 mL/g)")
        gas_imgs = sorted(glob.glob(os.path.join(OUT, "contour_*.png")))
        if gas_imgs:
            cols = st.columns(len(gas_imgs))
            for c, f in zip(cols, gas_imgs):
                g = os.path.basename(f).replace("contour_", "").replace(".png", "")
                c.image(PILImage.open(f), caption=g, use_container_width=True)

        st.markdown("#### Feasible settings")
        ok = grid[grid.feasible_robust] if grid.feasible_robust.any() else grid[grid.feasible_point]
        if ok.empty:
            st.warning("No setting meets every target. Lower the emission cut or yield target and recalculate.")
        else:
            rel_cols  = [c for c in ok.columns if c.endswith("_rel") and not c.endswith("_rel_hi")]
            show_cols = ["T", "t", "W", "SRY", "desirability"] + rel_cols
            st.dataframe(
                ok[show_cols].sort_values("desirability", ascending=False).head(20).round(3)
            )
            st.caption(
                "**T** = temperature (°C) · **t** = time (min) · **W** = water:dust (mL/g) · "
                "**SRY** = predicted yield (%) · **D** = desirability (0–1, higher is better) · "
                "***_rel** = emission as fraction of reference (lower is better)."
            )
        _dl(os.path.join(OUT, "operating_window.csv"),      "⬇  Feasible settings (.csv)")
        _dl(os.path.join(OUT, "operating_window_grid.csv"), "⬇  Full grid (.csv)")


# ══════════════════════════════════════════ PAGE: Q3
elif page_key == "q3":
    st.markdown('<div class="page-title">Q3 — Does activated carbon reduce emissions further?</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Compare the four carbon doses (0, 5, 10, 15%) within each '
        'washing condition and test whether the carbon effect is statistically significant.</div>',
        unsafe_allow_html=True,
    )

    wb = _get("wb_path"); runs = _get("runs")
    if not wb:
        st.warning("Upload your workbook on **Load Data** first."); st.stop()

    # ── dose–response plots
    st.markdown("### Dose–response")
    c1, c2 = st.columns([1, 3])
    gas         = c1.selectbox("Gas", pa.GASES)
    kind_choice = c1.radio("Response type", ["net_peak", "net_plateau"], horizontal=False)
    y = f"{gas}_{kind_choice}"

    if y in runs.columns and runs[y].notna().sum() > 0:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        means = runs.groupby("Cpct")[y].mean()
        axes[0].bar(means.index.astype(str), means.values, color="#2a9d8f", width=0.5,
                    edgecolor="white", linewidth=0.8)
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
        st.info(f"No data for {y} yet.")

    st.markdown("---")

    # ── carbon terms from mixed model
    st.markdown("### Carbon sub-plot model terms")
    em_results = _get("em_results")
    if em_results is None:
        st.info("Run the emission analysis on **Q1 Part B** first.")
    else:
        for key in [f"{gas}_net_peak", f"{gas}_net_plateau"]:
            if key not in em_results:
                continue
            mm = em_results[key]
            pv = mm.pvalues.drop([i for i in mm.pvalues.index if i == "Group Var"], errors="ignore")
            carbon_terms = pv[[i for i in pv.index if "Cc" in i or "carbon" in i.lower()]]
            if len(carbon_terms) == 0:
                st.write(f"No carbon terms in {key} model."); continue
            df_ct = pd.DataFrame({"coef": mm.params[carbon_terms.index],
                                   "p":    carbon_terms}).round(4)
            df_ct["significant"] = df_ct["p"] < 0.05
            st.markdown(f"**{key}**")
            st.dataframe(df_ct.style.map(
                lambda v: "background-color:#d4edda" if v else "", subset=["significant"]))

    st.markdown("---")

    # ── box plots all gases
    st.markdown("### Net peak by dose — all gases")
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

    # ── BH table
    bh_path = os.path.join(OUT, "bh_correction.csv")
    if os.path.exists(bh_path):
        st.markdown("### Multiple-testing correction")
        st.dataframe(pd.read_csv(bh_path, index_col=0).round(4))
        st.caption("BH-adjusted p-values control the false-discovery rate.")

    st.markdown("---")
    st.info(
        "**How to read the carbon terms**\n\n"
        "- **Cc** — linear dose effect (significant = more carbon changes emissions proportionally)\n"
        "- **I(Cc²)** — curved effect (diminishing returns or an optimum dose)\n"
        "- **Cc:x1 / Cc:x2 / Cc:x3** — the carbon effect depends on temperature / time / ratio\n\n"
        "PM2.5 and PM10 are excluded (sensor at ceiling). "
        "Simulated workbook results test the code only."
    )
