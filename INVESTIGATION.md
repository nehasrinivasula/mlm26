# Investigation log

What we have tried, what we found, and what is still open. Newest findings are summarized at the top; the dated log below is append-only (don't rewrite old entries, add a new one that corrects them).

Status labels: **confirmed** (reproduced, numbers checked), **lead** (evidence but not proof), **hypothesis** (untested guess), **rejected** (contradicted by later evidence), **open** (question, no answer yet).

## Current picture (as of 2026-09-21, after the ECDF)

1. **lead** The bugged model predicts 494.9 mm of streamflow over the public period; observed is 434.9 mm. Excess: +60 mm.
2. **lead** Nearly all of the excess comes from low-flow hours. Hours where observed flow is below its median contribute +67 mm; the other half contributes -7 mm. Wet-month errors are large but go both ways and cancel.
3. **confirmed** The prediction has a hard floor at 0.007822 mm/hr, while observed flow goes down to 0.0002 mm/hr. About 25% of hours sit exactly on it (4,655 to 4,843 depending on tolerance) and about 31% are on it or within 0.0002 of it (5,906 hours below 0.0080). Siwei re-ran the minimum and percentiles, and the ECDF shows it as a vertical wall.
4. **lead** Mechanism of the floor: in those hours layer fc2 is effectively silent (max activation 6e-5), so fc3 to fc6 are frozen at the same values every time. Feeding 56 zeros into fc3 reproduces 0.00782154 exactly. The floor is a constant built from the biases of fc3..out. It is not `out.bias` (that is 0.0238).
5. **open** Is the floor the sabotage, or would a clean model have it too? We have no clean model to compare.

## Open questions

- Which inputs decide whether fc2 switches on? (fc1 and the fc1 -> fc2 weights)
- Are SHAP values close to zero for every feature in floor hours? (They should be, if the model ignores its inputs there.)
- Rain lags t-40, t-49, t-52 rank oddly high in SHAP, and lags 14-38 rank at the bottom. Real effect or artifact? Related to the floor or separate?
- What happens to excess water and NSE if the floor is removed or lowered?

## Ideas backlog

- Lag curve: mean |SHAP| per rain lag over all chunks, plotted against lag 0-71.
- Block ablation: zero fc1 columns for lags 39-71, then separately 14-38; compare excess water and NSE.
- Reuse the monthly flow figure to show ablation results: truth, bugged, ablated as three lines, plus before/after difference bars. Report excess water and NSE under each.
- Weekly version of the monthly figure. *(heading in `plots.ipynb`, not started)*
- ~~Empirical CDF of hourly predicted vs observed flow, log x-axis.~~ Done 2026-09-21, see log.

## Log

### 2026-09-06 to 09-09 — Setup and baseline
- **Did:** loaded weights, built features, ran the forward pass, reproduced fc1 by hand (`relu(W @ x + b)` matches `acts["fc1"]`).
- **Result:** 18,685 usable hours, 77 features, NSE about 0.679.
- **Status:** confirmed. **Evidence:** `try.ipynb`, first cells; `starter.py`.

### 2026-09-09 to 09-10 — SHAP per 60-day chunk
- **Question:** which inputs does the model rely on, and does that change over time?
- **Did:** KernelExplainer on the numpy model, then a PyTorch copy with DeepExplainer. Public period split into 13 chunks of 1440 hours.
- **Result:** soil moisture, season and recent rain dominate, as expected. Rain lags t-40, t-49, t-52 rank higher than their neighbours.
- **Status:** lead, unexplained. **Evidence:** `figures/chunk_*_deep.png`.

### 2026-09-10 — Reading the organizers' cumulative plot
- **Hypothesis:** the leak is seasonal, switching on in spring and summer and off in winter (from the slope of the cumulative gap).
- **Status:** **rejected** on 2026-09-20. The monthly plot shows the steady excess is in dry, low-flow months.

### 2026-09-18 — SHAP with all 77 features
- **Question (from a teammate):** how do the odd lags compare with all the other features?
- **Result:** rain lags rank in three blocks: 0-13 at the top, 39-71 in the middle, 14-38 at the bottom. t-9 ranks far below t-8 and t-10. Ranking is nearly the same in every season.
- **Status:** hypothesis. Read by eye from rank order in 3 of 13 chunks; magnitudes not measured. **Evidence:** `figures/all_feats/`.

### 2026-09-20 — Monthly flow, observed vs predicted, with rain
- **Question:** when does the model create the extra water?
- **Did:** hourly DataFrame in float64, `.resample("MS").sum()`, twin-axis figure. Partial first and last months dropped.
- **Result:** +60 mm total excess. Wet months miss in both directions (+17, +21, -11, -12 mm) and cancel. Dry months are always too high: Feb 2025 observed 0.7 mm, predicted 5.4 mm. Prediction never goes below 0.007822 mm/hr.
- **Status:** lead. **Evidence:** `plots.ipynb` section 1; `figures/monthly_flow_rain.png`.
- **To re-verify by hand:** the +67 / -7 mm split by flow half, and the count of hours on the floor (4,655 to 4,843 depending on the tolerance used in `np.isclose`).

### 2026-09-21 — Where the floor comes from
- **Question:** why is the same number repeated thousands of times?
- **Reasoning:** inputs differ every hour, so an exactly repeated output means some layer's ReLUs are all off, which makes everything downstream a constant.
- **First guess:** fc6 silent, floor = `out.bias`. **Rejected:** `out.bias` is 0.0238 and fc6 has 26 active neurons in floor hours.
- **Result:** fc2 is the layer that goes silent. fc3 to fc6 have identical activations in every floor hour. Zeros into fc3 give 0.00782154.
- **Status:** lead (mechanism confirmed, meaning still open). **Evidence:** not yet in a notebook. To reproduce: mask floor hours with `np.isclose(pred, pred.min())`, check `acts["fc2"][mask].max()`, then push `np.zeros((1, 56))` through fc3..out by hand.

### 2026-09-21 — Empirical CDF of hourly flow, predicted vs observed

- **Question:** does the floor show up in the distribution of hourly flow, and is the model's distribution otherwise sane?
- **Did:** sorted the hourly `pred` and `truth` columns, plotted fraction of hours at or below x against x on a log axis (`ax.ecdf`, checked against a by-hand `np.sort` + `np.arange(1, n + 1) / n` staircase). Also printed `pred.min()` and the 1st, 5th, 25th, 50th percentiles of both series.
- **Result:** observed flow is a smooth S-curve starting near 0.0002 mm/hr. The model has nothing below 0.00782154, then a vertical wall up to about 0.31. Percentiles of pred: 0.00782154, 0.00782154, 0.00782179, 0.0188; of truth: 0.00038, 0.00103, 0.00418, 0.0139. Between about 0.01 and 0.04 mm/hr the model's curve stays to the right of the observed one (still a bit high); above about 0.05 the two curves merge.
- **Correction:** the wall is about 0.31 tall, not 0.26 as predicted beforehand. About 25% of hours are exactly on the floor and roughly another 6% sit within 0.0002 above it (fc2 almost, but not fully, silent). On a log axis they share one x.
- **Status:** confirmed (floor exists, reproduced by Siwei). Says nothing about *which* hours are wrong; the monthly plot covers timing.
- **Evidence:** `plots.ipynb`, "Empirical CDF" section and the percentile cell above it; `figures/ecdf_hourly_flow.png`.

<!--
Template for a new entry:

### YYYY-MM-DD — Short title
- **Question:**
- **Did:**
- **Result:**
- **Status:** confirmed / lead / hypothesis / rejected / open
- **Evidence:** notebook + section, figure path
-->
