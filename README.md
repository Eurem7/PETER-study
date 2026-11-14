# PETER Study — Statistical Analysis Pipeline

**Project title**: Effect of Hot-Water Washing and Activated Carbon on Formaldehyde and VOC Emissions from MDF Dust  
**Department**: Wood and Biomaterials Engineering, University of Ibadan  
**Supervisor**: Dr. Kolajo  
**Statistics collaborator**: Dr. Olu (400-level Statistics, University of Ibadan)

---

## What this project does

Medium-density fibreboard (MDF) dust contains residual urea-formaldehyde resin that releases HCHO, TVOC, CO and CO₂ when handled or burned. This study asks:

| # | Research question |
|---|-------------------|
| **Q1** | Does hot-water washing reduce formaldehyde and other gas emissions from MDF dust, and which washing factors (temperature, time, water:dust ratio) drive the reduction? |
| **Q2** | What is the **operating window** — the set of washing conditions that simultaneously keep material yield ≥ 80 % *and* cut emissions by at least 50 % for all gases? |
| **Q3** | Does adding activated carbon to the wash water reduce emissions further, and does the carbon effect depend on the washing conditions? |

---

## Experimental design

### Washing experiment (Part 1)
A **3³ full-factorial design** varying three factors at three levels each:

| Factor | Low (−1) | Centre (0) | High (+1) |
|--------|-----------|------------|-----------|
| Temperature (°C) | 50 | 75 | 100 |
| Time (min) | 30 | 60 | 90 |
| Water:dust ratio (mL/g) | 5 | 10 | 15 |

27 unique wash conditions; centre run S14 replicated three times for pure-error estimation.  
Response: **final dry mass Md (g)** after washing and drying (initial dust M₀ = 50 g each time).  
Derived response: **solid-residue yield SRY (%)** = Md / M₀ × 100.

### PETER emission runs (Part 2)
A **split-plot design**:
- **Whole-plot factor**: wash condition (the 27 combinations above, carried forward from Part 1)
- **Sub-plot factor**: activated-carbon dose at 4 levels — 0, 5, 10, 15 % of dust mass

Each wash generates 4 PETER chamber runs (one per carbon dose), giving up to 108 rows.  
Responses: **HCHO, TVOC, CO, CO₂** — each recorded as baseline, peak and plateau.  
Net peak = peak − baseline; net plateau = plateau − baseline.

> PM2.5 and PM10 are excluded from modelling because sensor readings are at the ceiling throughout all runs.

---

## Statistical methods

### Part 1 — Yield (Md)
1. **Second-order response surface model** (RSM) in coded variables x1, x2, x3  
   Full model: `Md ~ x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3 + I(x1²) + I(x2²) + I(x3²)`
2. Terms that are non-estimable with the actual design levels are dropped automatically and reported.
3. **Type II ANOVA** (sequential SS, each term adjusted for all others).
4. **Lack-of-fit vs pure-error** split when centre replicates exist.
5. **Box-Cox** power transform check (λ with 95 % CI; flags when raw scale is not recommended).
6. **Residual diagnostics**: Shapiro-Wilk (normality), Breusch-Pagan (heteroscedasticity), Cook's D > 4/n (influence).
7. **Backward elimination** (hierarchy preserved) produces a reduced model; model with higher predicted R² (PRESS) is carried forward.
8. **Model equation** exported in coded and natural-unit notation.

### Part 2 — Emissions
For each gas × response type (net peak, net plateau):

**(a) Linear mixed model** (wash = random intercept) for the full 108-row dataset.  
Fixed effects: all RSM terms for x1/x2/x3 + carbon terms Cc, Cc², Cc:x1, Cc:x2, Cc:x3 + optional day block.  
Optimiser fallback chain (lbfgs → powell → Nelder-Mead); day block dropped when singular.

**(b) Whole-plot F-test** on the 27 wash means — the **exact test** for whole-plot factors (temperature, time, ratio). RSM fitted to wash-mean response; backward elimination + ANOVA + LOF + diagnostics applied.

**(c) Tobit regression** (right-censored at the sensor ceiling) with **cluster-robust sandwich standard errors** (clustered by wash) for responses where ≥ 5 % of peaks hit the ceiling.

**(d) Benjamini-Hochberg FDR correction** across all gas × kind model F-tests to control the false-discovery rate from testing multiple responses.

### Operating window (Q2)
- **Grid search** over 21³ = 9261 combinations of T/t/W (coded −1 to +1 in steps of 0.1).
- Each grid point is classified **robust** (95 % lower prediction bound for yield also passes; 95 % upper emission bound also passes) or **point** (prediction alone passes).
- Required emission cut is `max(specified cut, 2 × pure-error SD / reference)` — the target is never smaller than the run-to-run noise floor.
- **Desirability function** D = geometric mean of yield desirability (0 below target, 1 at target + 10 %) and per-gas emission desirability (1 at zero, 0 at reference). The best D inside the feasible region is the **desirability optimum**.
- Outputs: feasible-settings table, window slice plots (3 water:dust levels), **desirability contour plot**, per-gas relative-emission contour plots.

---

## Repository layout

```
peter_analysis.py               Core analysis script (v3.0)
PETER_analysis.ipynb            Jupyter notebook (interactive, file dialog)
peter_app.py                    Streamlit web interface (4 pages)
requirements.txt                Python dependencies
README.md                       This file
.gitignore
PETER_Data_Workbook.xlsx        YOUR workbook (do not commit — in .gitignore)
PETER_Data_Workbook_SIMULATED_300.xlsx   Simulated test dataset
```

---

## Quick start

### Prerequisites
Python 3.9 or later. Install dependencies once:
```bash
pip install -r requirements.txt
```

### Option A — Streamlit web app *(recommended — answers Q1 → Q2 → Q3 in order)*
```bash
streamlit run peter_app.py
```
Open `http://localhost:8501` in your browser. Upload your workbook on the first page, then work through each question tab.

### Option B — Jupyter Notebook
```bash
jupyter notebook PETER_analysis.ipynb
```
Run cells top to bottom. A file dialog opens so you choose the workbook yourself. All plots appear inline.

### Option C — Script (command line)
```bash
python peter_analysis.py PETER_Data_Workbook.xlsx results/
```
All outputs are written to `results/`.

---

## Workbook requirements

The workbook must contain exactly these two sheets:

**Washing**

| Column | Description |
|--------|-------------|
| Run | Run number (integer) |
| Temp (C) | Wash temperature (°C) |
| Time (min) | Wash duration (min) |
| Water:dust (mL/g) | Water-to-dust ratio |
| Initial dust M0 (g) | Dust loaded (target 50 g) |
| Final dry mass Md (g) | Dry mass after washing |

**PETER_runs**

| Column | Description |
|--------|-------------|
| Wash run | Links to the Washing Run number |
| Temp (C) / Time (min) / Water:dust (mL/g) | Repeated from washing |
| Carbon (% of dust) | Activated-carbon dose (0, 5, 10, 15) |
| Date | Run date (used as a day block when >1 day) |
| HCHO baseline/peak/plateau | Raw sensor readings |
| TVOC baseline/peak/plateau | Raw sensor readings |
| CO baseline/peak/plateau | Raw sensor readings |
| CO2 baseline/peak/plateau | Raw sensor readings |

Columns may be blank (not yet measured) — the script skips any response with fewer than 20 non-missing values.

---

## Outputs written to `results/`

| File | Contents |
|------|----------|
| `part1_fit.png` | Observed vs fitted Md; residuals vs fitted |
| `diag_part1.png` | Q-Q plot, Cook's D (Part 1) |
| `part1_predictions.csv` | Observed and model-fitted Md for every wash |
| `equation_Md.txt` | Model equation for Md in coded + natural notation |
| `equation_{gas}_net_peak.txt` | Reduced whole-plot equation per gas |
| `diag_{gas}_net_peak.png` | Residual diagnostics per gas |
| `bh_correction.csv` | Benjamini-Hochberg FDR table across all gas models |
| `operating_window_grid.csv` | Full 9261-point grid with yield, emission and desirability |
| `operating_window.csv` | Feasible settings only |
| `window_by_ratio.png` | Window slices at 3 water:dust levels |
| `window_contour.png` | Desirability contour at W = 10 mL/g |
| `contour_{gas}.png` | Relative emission contour per gas at W = 10 mL/g |

---

## Streamlit app pages

| Page | Answers |
|------|---------|
| **Load Data** | Upload workbook; preview both sheets; missing-value table |
| **Q1: Does Washing Work?** | Part 1 yield RSM + diagnostics + equation; Part 2 emission models + BH table + diagnostic plots + equations |
| **Q2: Operating Window** | Adjustable sliders for dose, yield target, cut target; window plots + contour plots + desirability table |
| **Q3: Carbon Effect** | Dose–response plots, dose × temperature interaction, carbon sub-plot model terms, box plots across all gases |

---

## Limitations and what's still needed

| Item | Status |
|------|--------|
| Moisture readings (drying mass) | Columns blank — yield is as-weighed until provided |
| Real PETER emission data | Simulated data used for testing; replace with real readings |
| Confirmation run | Operating window is a model prediction — one lab run is needed to verify it |
| CO zero-inflation | ~10 % of CO readings may be zero; a hurdle/zero-inflated model would be more accurate |
| Joint confidence region for window | Current approach tests yield and emissions independently |
| Cross-check vs R/Minitab | Recommended before final write-up |
| PM2.5 / PM10 | Not modelled; ceiling-hitting throughout all runs |

---

## Citing this analysis

If you reference the statistical approach in the thesis:

> Response surface modelling was performed in Python 3 using `statsmodels` (Seabold & Perktold, 2010) with second-order RSM, split-plot mixed models, Tobit censored regression with cluster-robust standard errors, Benjamini-Hochberg FDR correction, and a desirability-based operating window. All code is available in the project repository.

---

## License

For academic use within this FYP. Contact Dr. Olu for reuse.
