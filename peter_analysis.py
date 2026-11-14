"""PETER study analysis pipeline.

Usage: python3 peter_analysis.py WORKBOOK.xlsx [OUTDIR]

Part 1  Washing   : second-order response surface for final dry mass Md (g)
                    (equivalently recovery, since M0 = 50 g), with ANOVA,
                    R2 / adj R2 / predicted R2 (PRESS) and lack of fit if
                    centre replicates exist.  Non-estimable terms are dropped
                    automatically and reported.
Part 2  PETER_runs: split-plot analysis for HCHO, TVOC, CO, CO2 (net peak and
                    plateau).  T, t, W are whole-plot factors (27 washes),
                    carbon C is the sub-plot factor.
                      (a) mixed model, wash = random intercept
                      (b) whole-plot test: wash-mean response ~ RSM(T,t,W)
                          (the exact F-test for whole-plot effects)
                      (c) Tobit (right-censored) for responses at a sensor ceiling
Window  : grid search + desirability optimum of washing conditions meeting yield
          and emission targets, with contour plots.
"""
__version__ = "3.0"

import sys, os, warnings, itertools, io, contextlib
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import optimize, stats
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

warnings.filterwarnings("ignore")

GASES     = ["HCHO", "TVOC", "CO", "CO2"]
CEILING   = {"PM2.5": 999, "PM10": 999, "CO2": 5000}
WASH_TERMS = "x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3 + I(x1**2) + I(x2**2) + I(x3**2)"


# ----------------------------------------------------------- column resolver
# Maps canonical internal names → all accepted surface forms.
# Resolution is case-insensitive and strips extra whitespace, so
# "Temp (°C)", "TEMP(C)", "Temperature" all resolve to the same column.

_ALIASES: dict[str, list[str]] = {
    # ── Washing / PETER_runs shared ──────────────────────────────────────
    "Temp (C)": [
        "temp (c)", "temperature (c)", "temp (°c)", "temperature (°c)",
        "temp(c)", "temperature", "temp", "t (°c)", "t (c)",
    ],
    "Time (min)": [
        "time (min)", "time(min)", "time (minutes)", "time", "duration (min)",
        "duration", "t (min)",
    ],
    "Water:dust (mL/g)": [
        "water:dust (ml/g)", "water:dust (ml/g)", "water:dust(ml/g)",
        "water:dust (ml/g)", "water : dust (ml/g)", "water/dust (ml/g)",
        "w:d (ml/g)", "water ratio (ml/g)", "water ratio", "ratio (ml/g)",
        "w (ml/g)", "water:dust",
    ],
    # ── Washing only ─────────────────────────────────────────────────────
    "Run": ["run", "run no", "run no.", "run number", "run #", "id"],
    "Initial dust M0 (g)": [
        "initial dust m0 (g)", "m0 (g)", "m0(g)", "initial mass (g)",
        "initial dust (g)", "initial weight (g)", "m0",
    ],
    "Final dry mass Md (g)": [
        "final dry mass md (g)", "md (g)", "md(g)", "final mass (g)",
        "dry mass (g)", "final dry mass (g)", "md",
    ],
    # ── PETER_runs only ──────────────────────────────────────────────────
    "Wash run": [
        "wash run", "wash", "wash no", "wash no.", "wash id",
        "washing run", "run (wash)", "wash run no",
    ],
    "Carbon (% of dust)": [
        "carbon (% of dust)", "carbon (%)", "carbon(%)", "carbon % of dust",
        "activated carbon (%)", "ac (%)", "c (%)", "carbon dose (%)",
        "carbon dose", "carbon addition (%)",
    ],
    "Date": ["date", "run date", "test date", "day", "experiment date"],
}

# Gas sensor columns: auto-generate aliases for HCHO baseline / peak / plateau etc.
for _g in GASES + ["PM2.5", "PM10"]:
    for _k in ("baseline", "peak", "plateau", "area"):
        _canonical = f"{_g} {_k}"
        _ALIASES[_canonical] = [
            f"{_g} {_k}",
            f"{_g.lower()} {_k}",
            f"{_g}{_k}",          # e.g. HCHOpeak
            f"{_g}_{_k}",         # e.g. HCHO_peak
            f"{_g} {_k} conc",
            f"{_g} {_k} (ppm)",
            f"{_g} {_k} (ppb)",
            f"{_g} {_k} reading",
        ]


def _normalise(s: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation clutter."""
    import re
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _resolve_columns(actual_cols: list[str], required: list[str]) -> tuple[dict[str, str], list[str]]:
    """Return (rename_map, unresolved_list).

    rename_map  : {actual_col_name: canonical_name}  for every required column found
    unresolved  : canonical names that couldn't be matched
    """
    norm_actual = {_normalise(c): c for c in actual_cols}
    rename_map: dict[str, str] = {}
    unresolved: list[str] = []

    for canon in required:
        # 1. exact match first
        if canon in actual_cols:
            rename_map[canon] = canon
            continue
        # 2. normalised exact match
        nc = _normalise(canon)
        if nc in norm_actual:
            rename_map[norm_actual[nc]] = canon
            continue
        # 3. alias list
        aliases = [_normalise(a) for a in _ALIASES.get(canon, [])]
        matched = next((norm_actual[a] for a in aliases if a in norm_actual), None)
        if matched:
            rename_map[matched] = canon
            continue
        # 4. substring heuristic: canonical words all appear somewhere in an actual col
        canon_words = set(_normalise(canon).split())
        candidates = [
            orig for norm, orig in norm_actual.items()
            if canon_words.issubset(set(norm.split()))
        ]
        if len(candidates) == 1:
            rename_map[candidates[0]] = canon
            continue
        unresolved.append(canon)

    return rename_map, unresolved


# ----------------------------------------------------------- input check
REQUIRED = {
    "Washing":    ["Run", "Temp (C)", "Time (min)", "Water:dust (mL/g)",
                   "Initial dust M0 (g)", "Final dry mass Md (g)"],
    "PETER_runs": ["Wash run", "Temp (C)", "Time (min)", "Water:dust (mL/g)",
                   "Carbon (% of dust)", "Date"]
                  + [f"{g} {k}" for g in GASES for k in ("baseline", "peak", "plateau")],
}

# Columns in PETER_runs that are allowed to be absent (sensor not yet measured)
_OPTIONAL = {f"{g} {k}" for g in GASES + ["PM2.5", "PM10"] for k in ("baseline", "peak", "plateau", "area")}


def check_workbook(path):
    """Fail early with a clear message if a sheet or required column is missing.

    Uses fuzzy matching so minor name variations are accepted automatically.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    xl = pd.ExcelFile(path)
    problems = []
    for sheet, required_cols in REQUIRED.items():
        if sheet not in xl.sheet_names:
            problems.append(
                f"Missing sheet '{sheet}' — found: {', '.join(xl.sheet_names)}")
            continue
        have = list(pd.read_excel(path, sheet_name=sheet, nrows=0).columns)
        _, unresolved = _resolve_columns(have, required_cols)
        # Gas sensor columns are optional (data may not be collected yet)
        hard_missing = [c for c in unresolved if c not in _OPTIONAL]
        if hard_missing:
            problems.append(
                f"Sheet '{sheet}' — cannot match these required columns: "
                + ", ".join(f"'{c}'" for c in hard_missing)
                + f"\n    Columns found: {', '.join(have)}"
            )
    if problems:
        raise ValueError(
            "Workbook layout problem — rename the highlighted columns "
            "and re-upload:\n\n  • " + "\n  • ".join(problems)
        )
    print(f"Workbook OK: {os.path.basename(path)}")


# ----------------------------------------------------------------- loading
def coded(df):
    df = df.copy()
    df["x1"] = (df["T"] - 75) / 25
    df["x2"] = (df["t"] - 60) / 30
    df["x3"] = (df["W"] - 10) / 5
    return df


def _apply_resolver(df: pd.DataFrame, required: list[str], sheet: str) -> pd.DataFrame:
    """Rename df columns to canonical names using the fuzzy resolver.

    Logs what was auto-mapped; raises ValueError for hard-missing columns.
    """
    rename_map, unresolved = _resolve_columns(list(df.columns), required)
    hard_missing = [c for c in unresolved if c not in _OPTIONAL]
    if hard_missing:
        raise ValueError(
            f"Sheet '{sheet}': cannot match required columns "
            + ", ".join(f"'{c}'" for c in hard_missing)
            + f"\n  Actual columns: {', '.join(df.columns)}"
        )
    # Report any non-trivial remaps so the user knows what was auto-matched
    non_trivial = {k: v for k, v in rename_map.items() if k != v}
    if non_trivial:
        print(f"  Auto-matched columns in '{sheet}':")
        for actual, canon in non_trivial.items():
            print(f"    '{actual}'  →  '{canon}'")
    return df.rename(columns=rename_map)


def load_washing(path):
    w = pd.read_excel(path, sheet_name="Washing")
    w = _apply_resolver(w, REQUIRED["Washing"], "Washing")
    w = w.rename(columns={"Temp (C)": "T", "Time (min)": "t", "Water:dust (mL/g)": "W",
                           "Initial dust M0 (g)": "M0", "Final dry mass Md (g)": "Md"})
    # Keep only the columns we need; ignore any extra columns in the sheet
    keep = [c for c in ["Run", "T", "t", "W", "M0", "Md"] if c in w.columns]
    w = w[keep]
    w["SRY"] = w["Md"] / w["M0"] * 100
    return coded(w)


def load_runs(path):
    r = pd.read_excel(path, sheet_name="PETER_runs")
    r = _apply_resolver(r, REQUIRED["PETER_runs"], "PETER_runs")
    r = r.rename(columns={"Wash run": "wash", "Temp (C)": "T", "Time (min)": "t",
                           "Water:dust (mL/g)": "W", "Carbon (% of dust)": "Cpct", "Date": "day"})
    for g in GASES + ["PM2.5", "PM10"]:
        for k in ("baseline", "peak", "plateau"):
            col = f"{g} {k}"
            if col in r:
                r[col] = pd.to_numeric(r[col], errors="coerce")
        if f"{g} peak" in r and f"{g} baseline" in r:
            r[f"{g}_net_peak"] = r[f"{g} peak"] - r[f"{g} baseline"]
        if f"{g} plateau" in r and f"{g} baseline" in r:
            r[f"{g}_net_plateau"] = r[f"{g} plateau"] - r[f"{g} baseline"]
        if f"{g} area" in r:
            r[f"{g}_area"] = pd.to_numeric(r[f"{g} area"], errors="coerce")
    r["wash"] = r["wash"].astype(int)
    r["Cc"] = (r["Cpct"] - 7.5) / 7.5
    return coded(r)


def load_controls(path):
    """Raw-dust control runs (untreated reference), if they have been measured."""
    try:
        c = pd.read_excel(path, sheet_name="Controls")
    except Exception:
        return pd.DataFrame()
    c = c[c["Type"].astype(str).str.contains("raw", case=False, na=False)].copy()
    for g in GASES:
        for k in ("baseline", "peak", "plateau"):
            col = f"{g} {k}"
            if col in c:
                c[col] = pd.to_numeric(c[col], errors="coerce")
        if f"{g} peak" in c and f"{g} baseline" in c:
            c[f"{g}_net_peak"] = c[f"{g} peak"] - c[f"{g} baseline"]
        if f"{g} plateau" in c and f"{g} baseline" in c:
            c[f"{g}_net_plateau"] = c[f"{g} plateau"] - c[f"{g} baseline"]
    return c


# ------------------------------------------------------------ RSM helpers
def estimable_terms(df, terms):
    """Drop terms until the design matrix is full column rank."""
    from patsy import dmatrix
    kept, dropped = list(terms), []
    while True:
        X = np.asarray(dmatrix(" + ".join(kept), df))
        if np.linalg.matrix_rank(X) == X.shape[1] or len(kept) == 1:
            return kept, dropped
        for cand in reversed(kept):
            trial = [k for k in kept if k != cand]
            Xt = np.asarray(dmatrix(" + ".join(trial), df))
            if np.linalg.matrix_rank(Xt) == np.linalg.matrix_rank(X):
                kept, dropped = trial, dropped + [cand]
                break
        else:
            return kept, dropped


def rsm(df, y, terms=WASH_TERMS.split(" + "), label=""):
    d = df.dropna(subset=[y])
    kept, dropped = estimable_terms(d, terms)
    m = smf.ols(f"{y} ~ {' + '.join(kept)}", d).fit()
    h = m.get_influence().hat_matrix_diag
    press = np.sum((m.resid / (1 - h)) ** 2)
    sst = np.sum((d[y] - d[y].mean()) ** 2)
    out = dict(n=int(m.nobs), p=int(m.df_model + 1), df_resid=int(m.df_resid),
               R2=m.rsquared, adjR2=m.rsquared_adj, predR2=1 - press / sst,
               F=m.fvalue, pF=m.f_pvalue, RMSE=np.sqrt(m.mse_resid), dropped=dropped)
    return m, out


def print_rsm(m, o, title):
    print(f"\n=== {title} ===")
    print(f"n={o['n']}  parameters={o['p']}  residual df={o['df_resid']}")
    if o["dropped"]:
        print("Not estimable with this design (dropped):", ", ".join(o["dropped"]))
    print(f"R2={o['R2']:.3f}  adjR2={o['adjR2']:.3f}  predR2(PRESS)={o['predR2']:.3f}  "
          f"RMSE={o['RMSE']:.3f}  model F={o['F']:.2f} (p={o['pF']:.4f})")
    print(m.summary().tables[1])


def model_equation(m, y_label, transform_lambda=None):
    """Return the model equation in natural units as a readable string.

    If transform_lambda is not None (Box-Cox), the equation is for the
    transformed response and a note is added.
    """
    coefs = m.params
    intercept = coefs.get("Intercept", 0)
    # Map coded variables to natural labels
    natural = {"x1": "T", "x2": "t", "x3": "W"}
    lines = [f"{y_label} = {intercept:+.4f}"]
    for name, val in coefs.items():
        if name == "Intercept" or abs(val) < 1e-12:
            continue
        label = name
        for cd, nat in natural.items():
            label = label.replace(cd, nat)
        label = (label.replace("I(T**2)", "T²").replace("I(t**2)", "t²")
                      .replace("I(W**2)", "W²").replace("I(x1**2)", "T²")
                      .replace("I(x2**2)", "t²").replace("I(x3**2)", "W²"))
        lines.append(f"  {val:+.4f} × {label}")
    if transform_lambda is not None:
        note = (f"  [NOTE: response is Box-Cox transformed (λ={transform_lambda:.3f}); "
                f"back-transform with y = (ŷ·λ+1)^(1/λ) for λ≠0]")
        lines.append(note)
    else:
        lines.append("  [coded: T=(temp-75)/25, t=(time-60)/30, W=(ratio-10)/5]")
    return "\n".join(lines)


# ----------------------------------- transformation check
def boxcox_check(df, y):
    """Box-Cox power transform check. Returns (lambda, transformed_series, note).

    Returns lambda=None if the response has non-positive values or too few points.
    """
    d = df[y].dropna()
    if len(d) < 8 or (d <= 0).any():
        return None, df[y], "Box-Cox skipped (too few points or non-positive values)"
    lam_range = np.linspace(-2, 2, 81)
    llf = []
    for lam in lam_range:
        yt = stats.boxcox(d, lmbda=lam)
        n = len(yt)
        s2 = np.var(yt, ddof=1)
        llf.append(-n / 2 * np.log(s2) + (lam - 1) * np.sum(np.log(d)))
    best_lam = lam_range[np.argmax(llf)]
    ci_mask  = np.array(llf) >= max(llf) - stats.chi2.ppf(0.95, 1) / 2
    ci_lo, ci_hi = lam_range[ci_mask].min(), lam_range[ci_mask].max()
    transformed, _ = stats.boxcox(d)
    note = (f"Box-Cox λ={best_lam:.3f} (95% CI [{ci_lo:.2f}, {ci_hi:.2f}]). "
            + ("λ=1 is in the CI — raw scale is fine." if ci_lo <= 1 <= ci_hi
               else f"Consider transforming (λ≠1 outside CI)."))
    print(f"  Box-Cox: {note}")
    return best_lam, pd.Series(transformed, index=d.index), note


# ----------------------------------- ANOVA, diagnostics, model reduction
def anova_report(m, d, y):
    """Type II ANOVA, lack-of-fit vs pure-error split when settings are replicated."""
    tab = sm.stats.anova_lm(m, typ=2).rename(columns={"sum_sq": "SS", "PR(>F)": "p"})
    tab["MS"] = tab["SS"] / tab["df"]
    print("ANOVA (Type II):")
    print(tab[["SS", "df", "MS", "F", "p"]].round(4).to_string())
    g = d.groupby(["x1", "x2", "x3"])[y]
    sspe = sum(((s - s.mean()) ** 2).sum() for _, s in g if len(s) > 1)
    dfpe = sum(len(s) - 1 for _, s in g if len(s) > 1)
    if dfpe == 0:
        print("No replicated settings: pure error and lack of fit cannot be estimated.")
        return dict(sd_pe=None, p_lof=None)
    sd_pe = float(np.sqrt(sspe / dfpe))
    sslof, dflof = m.ssr - sspe, int(m.df_resid) - dfpe
    if dflof <= 0:
        return dict(sd_pe=sd_pe, p_lof=None)
    F = (sslof / dflof) / (sspe / dfpe)
    p = 1 - stats.f.cdf(F, dflof, dfpe)
    verdict = ("lack of fit is significant: the model form may be too simple" if p < 0.05
               else "no evidence of lack of fit")
    print(f"Lack of fit: SS={sslof:.4g} (df {dflof}); pure error: SS={sspe:.4g} (df {dfpe}); "
          f"F={F:.2f}, p={p:.4f} -> {verdict}")
    print(f"Pure-error SD: {sd_pe:.4g}")
    return dict(sd_pe=sd_pe, p_lof=p)


def diagnostics(m, name, out=None):
    """Residual checks: normality, constant spread, influential points."""
    from statsmodels.stats.diagnostic import het_breuschpagan
    r = np.asarray(m.resid)
    p_sw = stats.shapiro(r)[1] if len(r) >= 3 else np.nan
    try:
        p_bp = het_breuschpagan(r, m.model.exog)[1]
    except Exception:
        p_bp = np.nan
    cook = m.get_influence().cooks_distance[0]
    n_big = int((cook > 4 / len(r)).sum())
    flags = []
    if p_sw < 0.05:
        flags.append("residuals depart from normality")
    if p_bp < 0.05:
        flags.append("spread changes with the fitted value")
    if n_big:
        flags.append(f"{n_big} influential point(s) (Cook's D > 4/n)")
    print(f"Diagnostics: Shapiro-Wilk p={p_sw:.3f}; Breusch-Pagan p={p_bp:.3f}; "
          + ("FLAGS: " + "; ".join(flags) if flags else "no problems flagged"))
    if out:
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
        ax[0].scatter(m.fittedvalues, r, c="#1f6f8b"); ax[0].axhline(0, c="k", lw=1)
        ax[0].set(xlabel="Fitted", ylabel="Residual", title="Residuals vs fitted")
        sm.qqplot(r, line="s", ax=ax[1], markerfacecolor="#1f6f8b", markeredgecolor="#1f6f8b")
        ax[1].set_title("Normal Q-Q")
        ax[2].bar(range(len(cook)), cook, color="#1f6f8b")
        ax[2].axhline(4 / len(r), c="r", lw=1, ls="--")
        ax[2].set(xlabel="Observation", ylabel="Cook's D", title="Influence")
        fig.suptitle(name); fig.tight_layout()
        fig.savefig(os.path.join(out, f"diag_{name}.png"), dpi=120); plt.close(fig)
    return dict(p_shapiro=p_sw, p_bp=p_bp, n_influential=n_big)


def reduce_model(df, y, terms=WASH_TERMS.split(" + "), alpha=0.05):
    """Backward elimination keeping hierarchy."""
    d = df.dropna(subset=[y])
    kept, _ = estimable_terms(d, list(terms))

    def needed(t, ks):
        return t in ("x1", "x2", "x3") and any(t in k and k != t for k in ks)

    while len(kept) > 1:
        m = smf.ols(f"{y} ~ {' + '.join(kept)}", d).fit()
        p = pd.Series(m.pvalues.values[1:], index=kept)
        cand = [t for t in kept if p[t] > alpha and not needed(t, kept)]
        if not cand:
            break
        kept.remove(max(cand, key=lambda t: p[t]))
    m, o = rsm(d, y, terms=kept)
    return m, kept, o


def show_reduction(kept, o_red, o_full):
    print(f"Reduced model keeps: {', '.join(kept) or 'intercept only'}")
    print(f"    adjR2 {o_full['adjR2']:.3f} -> {o_red['adjR2']:.3f};  "
          f"predR2 {o_full['predR2']:.3f} -> {o_red['predR2']:.3f}")


# ---------------------------------------- multiple-testing correction
def bh_correct(pvalues_dict, alpha=0.05):
    """Benjamini-Hochberg FDR correction across all responses.

    pvalues_dict: {label: p_value}
    Returns DataFrame with original p, BH-adjusted p, and whether it passes.
    """
    labels = list(pvalues_dict.keys())
    pv     = np.array([pvalues_dict[l] for l in labels])
    n      = len(pv)
    order  = np.argsort(pv)
    rank   = np.empty(n); rank[order] = np.arange(1, n + 1)
    adj    = np.minimum(1, pv * n / rank)
    # enforce monotonicity (Benjamini-Hochberg)
    adj_mono = adj.copy()
    for i in range(n - 2, -1, -1):
        adj_mono[order[i]] = min(adj_mono[order[i]], adj_mono[order[i + 1]])
    df = pd.DataFrame({"p_raw": pv, "p_BH": adj_mono, "sig_BH": adj_mono < alpha},
                      index=labels)
    return df.sort_values("p_BH")


# --------------------------------------------------------------- Part 1
def part1(wash, out):
    print("\n" + "#" * 70 + "\nPART 1: washing  Md (final dry mass, g)\n" + "#" * 70)
    d = wash.dropna(subset=["Md"])
    print(f"{len(d)} of {len(wash)} washes have Md; "
          f"temperature levels present: {sorted(d['T'].unique())}")

    # Box-Cox check
    lam, _, _ = boxcox_check(d, "Md")

    m, o = rsm(wash, "Md")
    print_rsm(m, o, "Second-order RSM for Md (full model)")
    anova_report(m, d, "Md")
    diagnostics(m, "part1", out)
    mr, kept, orr = reduce_model(wash, "Md")
    show_reduction(kept, orr, o)
    final, label = (mr, "reduced") if orr["predR2"] >= o["predR2"] else (m, "full")
    print(f"Model carried forward: {label} (higher predicted R2)")

    # Natural-unit equation
    eq = model_equation(final, "Md (g)", transform_lambda=None)
    print("\n--- Equation in coded variables ---")
    print(eq)
    eq_path = os.path.join(out, "equation_Md.txt")
    with open(eq_path, "w") as f:
        f.write(eq)

    wash = wash.copy()
    wash["Md_hat"] = final.predict(wash)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(d["Md"], final.fittedvalues, c="#1f6f8b")
    lo, hi = d["Md"].min(), d["Md"].max()
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    ax[0].set(xlabel="Observed Md (g)", ylabel="Fitted Md (g)", title="Observed vs fitted")
    ax[1].scatter(final.fittedvalues, final.resid, c="#1f6f8b")
    ax[1].axhline(0, c="k", lw=1)
    ax[1].set(xlabel="Fitted", ylabel="Residual", title="Residuals")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "part1_fit.png"), dpi=130); plt.close(fig)
    wash[["Run", "T", "t", "W", "Md", "Md_hat"]].to_csv(
        os.path.join(out, "part1_predictions.csv"), index=False)
    return final


# --------------------------------------------------------------- Tobit
def tobit_right(df, y, formula_rhs, ceiling, group="wash"):
    """Right-censored Tobit by ML with cluster-robust sandwich SEs."""
    from patsy import dmatrix
    from statsmodels.tools.numdiff import approx_hess, approx_fprime
    d = df.dropna(subset=[y])
    X = np.asarray(dmatrix(formula_rhs, d))
    yv = d[y].values
    cens = yv >= ceiling - 1e-9
    b0 = np.linalg.lstsq(X, yv, rcond=None)[0]
    s0 = np.log(np.std(yv - X @ b0) + 1e-6)

    def ll_i(theta):
        b, ls = theta[:-1], theta[-1]
        s = np.exp(ls)
        mu = X @ b
        return np.where(cens, stats.norm.logsf((ceiling - mu) / s),
                        stats.norm.logpdf((yv - mu) / s) - ls)

    nll = lambda th: -ll_i(th).sum()
    res = optimize.minimize(nll, np.r_[b0, s0], method="BFGS")
    if not res.success:
        res = optimize.minimize(nll, res.x, method="Nelder-Mead",
                                options=dict(maxiter=20000, xatol=1e-6, fatol=1e-8))
    th = res.x
    H  = approx_hess(th, nll)
    bread = np.linalg.pinv(H)
    S  = approx_fprime(th, ll_i, centered=True)
    cl = pd.factorize(d[group].values)[0]
    G  = np.zeros((cl.max() + 1, S.shape[1]))
    np.add.at(G, cl, S)
    V  = bread @ (G.T @ G) @ bread
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    names = dmatrix(formula_rhs, d).design_info.column_names
    tab = pd.DataFrame({"coef": th[:-1], "se": se[:-1]}, index=names)
    tab["z"] = tab["coef"] / tab["se"]
    tab["p"] = 2 * stats.norm.sf(abs(tab["z"]))
    return tab, cens.mean()


# --------------------------------------------------------------- Part 2
def part2(runs, out, wash_model=None):
    print("\n" + "#" * 70 + "\nPART 2: PETER emissions (split-plot)\n" + "#" * 70)
    fixed_wp = "x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3 + I(x1**2) + I(x2**2) + I(x3**2)"
    fixed_sp = "Cc + I(Cc**2) + Cc:x1 + Cc:x2 + Cc:x3"
    use_day  = "day" in runs and runs["day"].nunique() > 1
    results  = {}
    all_wp_pvalues = {}   # collect for BH correction

    for g in GASES:
        for kind in ("net_peak", "net_plateau", "area"):
            y = f"{g}_{kind}"
            if y not in runs or runs[y].notna().sum() < 20:
                continue
            d = runs.dropna(subset=[y]).copy()
            ceil = CEILING.get(g)
            frac = (d[f"{g} peak"] >= ceil - 1e-9).mean() if ceil and f"{g} peak" in d else 0
            print(f"\n--- {y}  (n={len(d)} runs, {runs['wash'].nunique()} washes) ---")
            if kind == "net_peak" and ceil and frac > 0:
                print(f"{frac:.0%} of peaks sit at the sensor ceiling ({ceil}): censored.")

            # Box-Cox check on response
            boxcox_check(d, y)

            base_rhs = f"{fixed_wp} + {fixed_sp}"
            specs = ([(base_rhs + " + C(day)", "with day block")] if use_day else []) + \
                    [(base_rhs, "no day term")]
            mm, note = None, ""
            for rhs, lbl in specs:
                for meth in ("lbfgs", "powell", "nm"):
                    try:
                        cand = smf.mixedlm(f"{y} ~ {rhs}", d, groups=d["wash"]).fit(reml=True, method=meth)
                        if np.isfinite(cand.llf):
                            mm, note = cand, lbl; break
                    except Exception:
                        continue
                if mm is not None:
                    break
            try:
                if mm is None:
                    raise RuntimeError("no optimiser converged")
                if use_day and note == "no day term":
                    print("    (day block dropped: singular)")
                vc  = float(mm.cov_re.iloc[0, 0])
                icc = vc / (vc + mm.scale)
                print(f"(a) Mixed model: wash var={vc:.4g}, residual={mm.scale:.4g}, ICC={icc:.2f}")
                sig = mm.pvalues.drop([i for i in mm.pvalues.index if i == "Group Var"], errors="ignore")
                sig = sig[sig < 0.05].round(4)
                print("    significant fixed terms (Wald p<0.05):",
                      ", ".join(f"{k} (p={v})" for k, v in sig.items() if k != "Intercept") or "none")
                results[y] = mm
            except Exception as e:
                print("    mixed model failed:", e)

            # (b) whole-plot test on wash means
            wm = d.groupby("wash").agg({y: "mean", "x1": "first", "x2": "first",
                                         "x3": "first"}).reset_index()
            m_wp, o_wp = rsm(wm, y)
            print(f"(b) Whole-plot (wash means, n={len(wm)}): R2={o_wp['R2']:.3f} "
                  f"adjR2={o_wp['adjR2']:.3f} predR2={o_wp['predR2']:.3f}  F p={o_wp['pF']:.4f}"
                  + (f"  [dropped: {', '.join(o_wp['dropped'])}]" if o_wp["dropped"] else ""))
            coefs = m_wp.pvalues.drop("Intercept")
            print("    whole-plot terms p<0.05:",
                  ", ".join(f"{k} (p={v:.4f})" for k, v in coefs[coefs < 0.05].items()) or "none")
            if kind == "net_peak":
                anova_report(m_wp, wm, y)
                all_wp_pvalues[y] = o_wp["pF"]    # collect model F p-value for BH
            diagnostics(m_wp, y, out if kind == "net_peak" else None)
            mr_wp, kept_wp, orr_wp = reduce_model(wm, y)
            show_reduction(kept_wp, orr_wp, o_wp)

            # Natural-unit equation for the reduced whole-plot model
            eq = model_equation(mr_wp, y)
            with open(os.path.join(out, f"equation_{y}.txt"), "w") as f:
                f.write(eq)

            # (c) Tobit if censored
            if kind == "net_peak" and ceil and frac > 0 and f"{g} peak" in d:
                d["_y"] = d[f"{g} peak"]
                tab, cf = tobit_right(d, "_y", f"{fixed_wp} + {fixed_sp}", ceil)
                print(f"(c) Tobit on raw peak (right-censored at {ceil}, {cf:.0%} censored):")
                print(tab[tab["p"] < 0.05].round(4).to_string() or "    nothing significant")

    # BH correction across all whole-plot model F-tests
    if all_wp_pvalues:
        print("\n--- Benjamini-Hochberg FDR correction across whole-plot model F-tests ---")
        bh = bh_correct(all_wp_pvalues)
        print(bh.to_string())
        bh.to_csv(os.path.join(out, "bh_correction.csv"))
    return results


# --------------------------------------------------------------- Window
def _desirability_score(grid, gas_cols, yield_col="SRY", yield_min=80.0, ref_dict=None):
    """Overall desirability D = geometric mean of individual desirabilities.

    Yield desirability: 0 below yield_min, rises linearly to 1 at yield_min+10.
    Emission desirability: 1 at 0, falls linearly to 0 at the reference value.
    """
    d_yield = np.clip((grid[yield_col] - yield_min) / 10.0, 0, 1)
    d_em = {}
    for col in gas_cols:
        ref = ref_dict.get(col, grid[col].max()) if ref_dict else grid[col].max()
        d_em[col] = np.clip(1 - grid[col] / max(ref, 1e-9), 0, 1)
    all_d = [d_yield] + list(d_em.values())
    D = np.ones(len(grid))
    for di in all_d:
        D *= np.asarray(di)
    D = D ** (1.0 / len(all_d))
    return D


def window(wash_model, runs, out, controls=None, dose=10, reduction=0.5, yield_min=80.0,
           alpha=0.05, responses=GASES):
    """Operating window: grid search + desirability optimum + contour plots."""
    if wash_model is None:
        print("\nWindow skipped: no yield model."); return None
    dd = runs[runs["Cpct"] == dose]
    if dd["wash"].nunique() < 20:
        print(f"\nWindow skipped: fewer than 20 washes have runs at {dose}% carbon."); return None

    print("\n" + "#" * 70 +
          f"\nOPERATING WINDOW (carbon {dose}%, yield >= {yield_min}%, cut >= {reduction:.0%})\n" +
          "#" * 70)

    grid = pd.DataFrame(list(itertools.product(np.linspace(-1, 1, 21), repeat=3)),
                        columns=["x1", "x2", "x3"])
    z    = stats.norm.ppf(1 - alpha)
    py   = wash_model.get_prediction(grid).summary_frame(alpha=2 * alpha)
    grid["SRY"]    = py["mean"] / 50 * 100
    grid["SRY_lo"] = py["mean_ci_lower"] / 50 * 100
    point  = grid["SRY"] >= yield_min
    robust = grid["SRY_lo"] >= yield_min

    used = []
    ref_dict_for_des = {}
    for g in responses:
        y    = f"{g}_net_peak"
        ceil = CEILING.get(g)
        if ceil and f"{g} peak" in dd and (dd[f"{g} peak"] >= ceil - 1e-9).mean() > 0.05:
            y = f"{g}_net_plateau"
            print(f"{g}: >5% at ceiling, using plateau instead.")
        if y not in dd or dd[y].notna().sum() < 20:
            print(f"{g}: skipped (not enough readings)."); continue
        w1 = dd.groupby("wash").agg({y: "mean", "x1": "first", "x2": "first",
                                      "x3": "first"}).reset_index()
        em, kept, _ = reduce_model(w1, y)
        sd_pe = None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                sd_pe = anova_report(em, w1.dropna(subset=[y]), y)["sd_pe"]
        except Exception:
            pass
        # reference
        c0 = runs[runs["Cpct"] == 0].groupby("wash").agg(
            {y: "mean", "x1": "first", "x2": "first", "x3": "first"}).reset_index()
        if controls is not None and len(controls) and y in controls and controls[y].notna().any():
            ref, src = float(controls[y].mean()), "raw-dust controls"
        elif len(c0) >= 20:
            e0, _, _ = reduce_model(c0, y)
            ref = float(e0.predict(pd.DataFrame({"x1": [-1], "x2": [-1], "x3": [-1]}))[0])
            src = "least-washed corner, no carbon"
        else:
            print(f"{g}: skipped (no reference)."); continue
        if not ref > 0:
            print(f"{g}: skipped (non-positive reference)."); continue
        need = reduction if sd_pe is None else min(0.95, max(reduction, 2 * sd_pe / ref))
        pe   = em.get_prediction(grid[["x1", "x2", "x3"]]).summary_frame(alpha=2 * alpha)
        grid[f"{g}_rel"]    = pe["mean"] / ref
        grid[f"{g}_rel_hi"] = pe["mean_ci_upper"] / ref
        point  &= grid[f"{g}_rel"]    <= 1 - need
        robust &= grid[f"{g}_rel_hi"] <= 1 - need
        ref_dict_for_des[f"{g}_rel"] = 1.0
        used.append(g)
        print(f"{g}: ref {ref:.4g} ({src}); cut {need:.0%}"
              + (f" (noise floor, pe SD={sd_pe:.3g})" if sd_pe else "")
              + f"; terms: {', '.join(kept) or 'intercept'}")

    if not used:
        print("Window skipped: no gas could be modelled."); return None

    grid["T"] = 75 + 25 * grid.x1
    grid["t"] = 60 + 30 * grid.x2
    grid["W"] = 10 + 5  * grid.x3
    grid["feasible_point"]  = point
    grid["feasible_robust"] = robust

    # Desirability score
    gas_cols = [f"{g}_rel" for g in used if f"{g}_rel" in grid]
    grid["desirability"] = _desirability_score(grid, gas_cols, ref_dict=ref_dict_for_des,
                                               yield_min=yield_min)
    grid.to_csv(os.path.join(out, "operating_window_grid.csv"), index=False)

    n_pt = int(point.sum()); n_rb = int(robust.sum())
    print(f"\nFeasible: {n_pt} point-prediction, {n_rb} robust (of {len(grid)} grid points).")

    ok  = grid[robust] if robust.any() else grid[point]
    tag = "robust" if robust.any() else "point-prediction only"
    if len(ok):
        ok.to_csv(os.path.join(out, "operating_window.csv"), index=False)
        print(f"Window ({tag}): " + "; ".join(
            f"{c} {ok[c].min():.4g}–{ok[c].max():.4g}" for c in ("T", "t", "W")))
        best = ok.loc[ok["desirability"].idxmax()]
        print(f"Desirability optimum: T={best['T']:.0f} °C, t={best['t']:.0f} min, "
              f"W={best['W']:.1f} mL/g; yield {best['SRY']:.1f}%; "
              f"desirability D={best['desirability']:.3f}; "
              + ", ".join(f"{g} at {best[g+'_rel']:.0%} of ref" for g in used))
        print("This is a model prediction — confirm with a lab run.")
    else:
        print("No setting meets every target. Relax REDUCTION or YIELD_MIN.")

    # --- figures: (1) window slices, (2) contour plots ---
    ratios = [-1, 0, 1]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, x3v in zip(axes, ratios):
        s_ = grid[np.isclose(grid.x3, x3v)]
        ax.scatter(s_["T"], s_["t"], c="#d9d9d9", s=26, label="not feasible")
        sp = s_[s_.feasible_point & ~s_.feasible_robust]
        ax.scatter(sp["T"], sp["t"], c="#f2b134", s=26, label="point only")
        sr = s_[s_.feasible_robust]
        ax.scatter(sr["T"], sr["t"], c="#2a9d8f", s=26, label="robust")
        ax.set(xlabel="Temperature (°C)", title=f"Water:dust = {10 + 5 * x3v:.0f} mL/g")
    axes[0].set_ylabel("Time (min)"); axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Operating window (carbon {dose}%)"); fig.tight_layout()
    fig.savefig(os.path.join(out, "window_by_ratio.png"), dpi=130); plt.close(fig)

    # Desirability contour at middle ratio (x3=0)
    s_mid = grid[np.isclose(grid.x3, 0)].copy()
    if len(s_mid) >= 4:
        T_vals = np.sort(s_mid["T"].unique())
        t_vals = np.sort(s_mid["t"].unique())
        Z      = s_mid.pivot(index="t", columns="T", values="desirability").values
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        cf = ax2.contourf(T_vals, t_vals, Z, levels=10, cmap="YlGn")
        plt.colorbar(cf, ax=ax2, label="Desirability D")
        if robust.any():
            sr = s_mid[s_mid.feasible_robust]
            ax2.scatter(sr["T"], sr["t"], c="white", edgecolors="#2a9d8f", s=22, label="robust feasible")
        ax2.set(xlabel="Temperature (°C)", ylabel="Time (min)",
                title=f"Desirability contour (W=10 mL/g, carbon {dose}%)")
        ax2.legend(fontsize=8); fig2.tight_layout()
        fig2.savefig(os.path.join(out, "window_contour.png"), dpi=130); plt.close(fig2)

    # Contour for each gas (relative emission at x3=0)
    for g in used:
        col = f"{g}_rel"
        if col not in s_mid:
            continue
        Z_g = s_mid.pivot(index="t", columns="T", values=col).values
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        cf3 = ax3.contourf(T_vals, t_vals, Z_g, levels=10, cmap="RdYlGn_r")
        plt.colorbar(cf3, ax=ax3, label=f"{g} relative to reference")
        ax3.set(xlabel="Temperature (°C)", ylabel="Time (min)",
                title=f"{g} relative emission (W=10 mL/g, carbon {dose}%)")
        fig3.tight_layout()
        fig3.savefig(os.path.join(out, f"contour_{g}.png"), dpi=130); plt.close(fig3)

    return grid


def main():
    matplotlib.use("Agg")
    path = sys.argv[1]
    check_workbook(path)
    out = sys.argv[2] if len(sys.argv) > 2 else "peter_out"
    os.makedirs(out, exist_ok=True)
    wash = load_washing(path)
    runs = load_runs(path)
    wm   = part1(wash, out)
    part2(runs, out, wm)
    window(wm, runs, out, controls=load_controls(path))


if __name__ == "__main__":
    main()
