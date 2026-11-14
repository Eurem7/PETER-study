"""PETER Study — Streamlit Web Interface
Run:  streamlit run peter_app.py

The app answers the three project questions in order:
  Q1 Does hot-water washing reduce emissions? Which factors matter?
  Q2 What is the operating window (yield ≥ 80%, low emissions)?
  Q3 Does activated carbon reduce emissions further?
"""

import os, sys, pathlib, importlib.util, re, io, contextlib, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image as PILImage

# ───────────────────────────────────────────────────────── page config
st.set_page_config(
    page_title="PETER Study Analysis",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────────────────────────────────── load pa module
@st.cache_resource
def load_pa():
    here = pathlib.Path(__file__).parent
    script = here / "peter_analysis.py"
    if not script.exists():
        st.error(f"peter_analysis.py not found in {here}. Place it next to this file.")
        st.stop()
    text = script.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    ver = m.group(1) if m else "unknown"
    sys.modules.pop("peter_analysis", None)
    spec = importlib.util.spec_from_file_location("peter_analysis", script)
    pa = importlib.util.module_from_spec(spec)
    sys.modules["peter_analysis"] = pa
    spec.loader.exec_module(pa)
    import matplotlib
    matplotlib.use("Agg")
    return pa, ver

pa, pa_ver = load_pa()

# ───────────────────────────────────────────────────────── sidebar nav
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/8/8c/University_of_Ibadan_logo.png",
                 width=80, use_container_width=False)
st.sidebar.title("PETER Study")
st.sidebar.caption(f"Analysis pipeline v{pa_ver}")

pages = {
    "🏠 Load Data":          "load",
    "Q1: Does Washing Work?": "q1",
    "Q2: Operating Window":   "q2",
    "Q3: Carbon Effect":      "q3",
}
page = st.sidebar.radio("Navigate", list(pages.keys()), label_visibility="collapsed")
page_key = pages[page]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How to use**\n"
    "1. Upload your workbook on **Load Data**\n"
    "2. Work through Q1 → Q2 → Q3\n"
    "3. Download results from each page\n\n"
    "*Simulated data only tests the code.*"
)

# ───────────────────────────────────────────────────────── session helpers
OUT = "results"
os.makedirs(OUT, exist_ok=True)

def _get(key, default=None):
    return st.session_state.get(key, default)

def _set(key, val):
    st.session_state[key] = val

def _img(path, caption=""):
    if os.path.exists(path):
        st.image(PILImage.open(path), caption=caption, use_container_width=True)

def _dl(path, label="Download"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            st.download_button(label, f.read(), file_name=os.path.basename(path))

def capture(func, *args, **kwargs):
    """Run func(*args, **kwargs), capture stdout, return (result, text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = func(*args, **kwargs)
    return result, buf.getvalue()


# ══════════════════════════════════════════════════════════ PAGE: LOAD DATA
if page_key == "load":
    st.title("🏠 Load your workbook")
    st.markdown(
        "Upload the **PETER_Data_Workbook.xlsx** (or the simulated version). "
        "The analysis on the other pages uses this file."
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
            _set("wash", wash)
            _set("runs", runs)
            st.success(f"✅ Loaded: **{uploaded.name}**")
        except Exception as e:
            st.error(f"Problem reading workbook: {e}")
            st.stop()

    wb = _get("wb_path")
    if wb:
        wash = _get("wash")
        runs = _get("runs")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Washing sheet")
            st.write(f"{len(wash)} rows | {wash['Md'].notna().sum()} have Md")
            st.dataframe(wash[["Run", "T", "t", "W", "Md", "SRY"]].round(2), height=250)
        with col2:
            st.subheader("PETER_runs sheet")
            st.write(f"{len(runs)} rows | {runs['wash'].nunique()} washes × "
                     f"{runs['Cpct'].nunique()} carbon doses")
            st.dataframe(runs[["wash", "T", "t", "W", "Cpct",
                                "HCHO_net_peak", "TVOC_net_peak",
                                "CO_net_peak", "CO2_net_peak"]].head(20).round(2), height=250)

        st.subheader("Missing values in key columns")
        cols = ["Md"] + [c for c in runs.columns if c.endswith(("net_peak", "net_plateau"))]
        mv   = pd.concat([
            wash[["Md"]].isna().sum().rename("Washing"),
            runs[[c for c in cols[1:] if c in runs]].isna().sum().rename("PETER_runs")
        ], axis=1).fillna("")
        st.dataframe(mv)
    else:
        st.info("Upload a workbook above to begin.")


# ══════════════════════════════════════════════════════════ PAGE: Q1
elif page_key == "q1":
    st.title("Q1: Does hot-water washing reduce formaldehyde emissions?")
    st.markdown(
        "This page fits a **second-order response-surface model** for the final dry mass "
        "(yield) after washing, then runs the **split-plot emission analysis** to show which "
        "washing factors (temperature, time, water ratio) drive HCHO, TVOC, CO and CO₂."
    )

    wb = _get("wb_path")
    if not wb:
        st.warning("Please upload your workbook on the **Load Data** page first.")
        st.stop()

    wash = _get("wash")
    runs = _get("runs")

    # ---- Part 1: yield model
    st.header("Part 1: Yield model (Md)")
    if st.button("Run yield model (Part 1)", type="primary"):
        with st.spinner("Fitting response surface for Md…"):
            wm, out1 = capture(pa.part1, wash, OUT)
        _set("wash_model", wm)
        _set("part1_log", out1)
        st.success("Done.")

    if _get("wash_model"):
        with st.expander("Model output", expanded=True):
            st.code(_get("part1_log"), language="")
        col1, col2 = st.columns(2)
        with col1:
            _img(os.path.join(OUT, "part1_fit.png"), "Observed vs fitted Md")
        with col2:
            _img(os.path.join(OUT, "diag_part1.png"), "Residual diagnostics")

        eq_path = os.path.join(OUT, "equation_Md.txt")
        if os.path.exists(eq_path):
            with st.expander("📐 Model equation"):
                st.code(open(eq_path).read())
            _dl(eq_path, "Download equation (txt)")
        _dl(os.path.join(OUT, "part1_predictions.csv"), "Download predictions (csv)")

    # ---- Part 2: emissions
    st.header("Part 2: Emission analysis")
    st.markdown(
        "Uses a **split-plot mixed model** (wash = random intercept) for each gas. "
        "Temperature, time and water ratio are whole-plot factors; "
        "carbon dose is the sub-plot factor."
    )
    if _get("wash_model") is None:
        st.info("Run the yield model above first.")
    elif st.button("Run emission analysis (Part 2)", type="primary"):
        with st.spinner("Fitting split-plot models for all four gases…"):
            em_results, out2 = capture(pa.part2, runs, OUT, _get("wash_model"))
        _set("em_results", em_results)
        _set("part2_log", out2)
        st.success("Done.")

    if _get("part2_log"):
        with st.expander("Emission model output", expanded=False):
            st.code(_get("part2_log"), language="")

        bh_path = os.path.join(OUT, "bh_correction.csv")
        if os.path.exists(bh_path):
            st.subheader("Multiple-testing correction (Benjamini-Hochberg FDR)")
            st.dataframe(pd.read_csv(bh_path, index_col=0).round(4))
            st.caption("BH controls the false-discovery rate across all gas×kind model F-tests.")
            _dl(bh_path, "Download BH table (csv)")

        st.subheader("Residual diagnostic plots")
        for f in sorted(glob.glob(os.path.join(OUT, "diag_*_net_peak.png"))):
            gas = os.path.basename(f).replace("diag_", "").replace("_net_peak.png", "")
            _img(f, f"{gas} net peak — residuals")

        # Equations
        st.subheader("Reduced model equations")
        for f in sorted(glob.glob(os.path.join(OUT, "equation_*_net_peak.txt"))):
            gas = os.path.basename(f).replace("equation_", "").replace("_net_peak.txt", "")
            with st.expander(f"📐 {gas} equation"):
                st.code(open(f).read())


# ══════════════════════════════════════════════════════════ PAGE: Q2
elif page_key == "q2":
    st.title("Q2: What is the operating window?")
    st.markdown(
        "Find washing conditions that give **yield ≥ target** AND cut emissions by at least "
        "the required fraction for all four gases. Adjust the sliders and press Run."
    )

    wb = _get("wb_path")
    wm = _get("wash_model")
    runs = _get("runs")
    if not wb:
        st.warning("Upload your workbook on **Load Data** first.")
        st.stop()
    if wm is None:
        st.warning("Run the yield model on **Q1** first.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    dose       = col1.selectbox("Carbon dose (%)", [0, 5, 10, 15], index=2)
    yield_min  = col2.slider("Minimum yield (%)", 50.0, 95.0, 80.0, 1.0)
    reduction  = col3.slider("Required emission cut", 0.10, 0.90, 0.50, 0.05,
                              format="%.0f%%",
                              help="Fraction reduction vs the reference (0% carbon, least-washed corner).")

    if st.button("Calculate operating window", type="primary"):
        with st.spinner("Searching the grid…"):
            ctrl = pa.load_controls(wb)
            grid, out_w = capture(pa.window, wm, runs, OUT, controls=ctrl, dose=dose,
                                  reduction=reduction, yield_min=yield_min)
        _set("window_grid", grid)
        _set("window_log", out_w)
        st.success("Done.")

    if _get("window_log"):
        with st.expander("Window calculation output", expanded=True):
            st.code(_get("window_log"), language="")

    if _get("window_grid") is not None:
        grid = _get("window_grid")
        col1, col2 = st.columns(2)
        with col1:
            _img(os.path.join(OUT, "window_by_ratio.png"),
                 "Operating window (green = robust, amber = point only)")
        with col2:
            _img(os.path.join(OUT, "window_contour.png"),
                 "Desirability contour at W = 10 mL/g")

        st.subheader("Per-gas emission contours (W = 10 mL/g)")
        gas_imgs = sorted(glob.glob(os.path.join(OUT, "contour_*.png")))
        if gas_imgs:
            cols = st.columns(len(gas_imgs))
            for c, f in zip(cols, gas_imgs):
                g = os.path.basename(f).replace("contour_", "").replace(".png", "")
                c.image(PILImage.open(f), caption=g, use_container_width=True)

        st.subheader("Feasible settings table")
        ok = grid[grid.feasible_robust] if grid.feasible_robust.any() else grid[grid.feasible_point]
        if ok.empty:
            st.warning("No setting meets every target. Lower the emission cut or yield target.")
        else:
            show_cols = ["T", "t", "W", "SRY", "desirability"] + \
                        [c for c in ok.columns if c.endswith("_rel") and not c.endswith("_rel_hi")]
            st.dataframe(ok[show_cols].sort_values("desirability", ascending=False)
                           .head(20).round(3))
            st.caption(
                "Green rows = robust (95% confidence bound also passes). "
                "**Desirability D** = geometric mean of yield and emission desirabilities (1 = perfect)."
            )
        _dl(os.path.join(OUT, "operating_window.csv"), "Download feasible settings (csv)")
        _dl(os.path.join(OUT, "operating_window_grid.csv"), "Download full grid (csv)")


# ══════════════════════════════════════════════════════════ PAGE: Q3
elif page_key == "q3":
    st.title("Q3: Does activated carbon reduce emissions further?")
    st.markdown(
        "Compare emission levels across the four carbon doses (0, 5, 10, 15%) **within** each "
        "set of washing conditions. The split-plot model (Part 2) already estimates the carbon "
        "sub-plot terms. Here we visualise them directly."
    )

    wb   = _get("wb_path")
    runs = _get("runs")
    if not wb:
        st.warning("Upload your workbook on **Load Data** first.")
        st.stop()

    # ---- dose–response plots
    st.header("Dose–response by gas")
    gas = st.selectbox("Select gas", pa.GASES)
    kind_choice = st.radio("Response", ["net_peak", "net_plateau"], horizontal=True)
    y = f"{gas}_{kind_choice}"

    if y in runs.columns and runs[y].notna().sum() > 0:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        # mean by dose
        means = runs.groupby("Cpct")[y].mean()
        axes[0].bar(means.index.astype(str), means.values, color="#2a9d8f", width=0.5)
        axes[0].set(xlabel="Carbon dose (%)", ylabel=f"{gas} {kind_choice} (mean)",
                    title="Mean by carbon dose")
        # dose × temperature interaction
        for T, g_df in runs.groupby("T"):
            m = g_df.groupby("Cpct")[y].mean()
            axes[1].plot(m.index, m.values, marker="o", label=f"{T} °C")
        axes[1].set(xlabel="Carbon dose (%)", ylabel=f"{gas} {kind_choice}",
                    title="Dose × temperature interaction")
        axes[1].legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)
    else:
        st.info(f"No data for {y} yet.")

    # ---- carbon sub-plot model from Part 2
    st.header("Carbon sub-plot model terms")
    em_results = _get("em_results")
    if em_results is None:
        st.info("Run the emission analysis on **Q1** first to see significance.")
    else:
        for key in [f"{gas}_net_peak", f"{gas}_net_plateau"]:
            if key not in em_results:
                continue
            mm = em_results[key]
            pv = mm.pvalues.drop([i for i in mm.pvalues.index if i == "Group Var"], errors="ignore")
            carbon_terms = pv[[i for i in pv.index if "Cc" in i or "carbon" in i.lower()]]
            if len(carbon_terms) == 0:
                st.write(f"No carbon terms in {key} model.")
                continue
            df_ct = pd.DataFrame({"coef": mm.params[carbon_terms.index],
                                   "p":    carbon_terms}).round(4)
            df_ct["significant"] = df_ct["p"] < 0.05
            st.subheader(f"{key} — carbon terms")
            st.dataframe(df_ct.style.applymap(
                lambda v: "background-color: #d4edda" if v else "", subset=["significant"]))

    # ---- box plots across all gases
    st.header("Emissions at each carbon dose — box plots")
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    for ax, g in zip(axes2.flat, pa.GASES):
        y2 = f"{g}_net_peak"
        if y2 not in runs or runs[y2].isna().all():
            ax.set_visible(False); continue
        runs.boxplot(column=y2, by="Cpct", ax=ax)
        ax.set(title=g, xlabel="Carbon dose (%)", ylabel="Net peak")
        plt.sca(ax); plt.title(g)
    plt.suptitle("Net peak by carbon dose", y=1.01)
    plt.tight_layout()
    st.pyplot(fig2); plt.close(fig2)

    # ---- BH table (from Part 2)
    bh_path = os.path.join(OUT, "bh_correction.csv")
    if os.path.exists(bh_path):
        st.header("Multiple-testing correction across all gas models")
        st.dataframe(pd.read_csv(bh_path, index_col=0).round(4))
        st.caption("BH-adjusted p-values control the false-discovery rate.")

    st.markdown("---")
    st.info(
        "**Interpretation guide**\n\n"
        "- A significant `Cc` coefficient means carbon dose has a **linear** effect.\n"
        "- A significant `I(Cc**2)` coefficient means the effect is **curved** "
        "(e.g. diminishing returns).\n"
        "- `Cc:x1`, `Cc:x2`, `Cc:x3` terms mean the carbon effect **depends on** "
        "temperature, time or water ratio respectively.\n"
        "- PM2.5 and PM10 are not modelled (sensor ceiling).\n"
        "- Results from a **simulated** workbook test the code only."
    )
