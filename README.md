# mlm26 — Fixing the Poisoned Well

Team repo for the Kaggle challenge. This README is just a map of what's in here and how to get it running. For the challenge itself see the Kaggle page.

## Folder layout

```
.
├── data/
│   └── train.csv                 public period: hourly weather + soil inputs + observed streamflow (18,756 rows)
├── model/
│   ├── streamflow_model_bug.npz  the poisoned model weights (fc1..fc6, out; .weight and .bias each)
│   ├── feature_scaler.json       z-score mean/std for the 77 inputs (fit on the FULL dataset incl. held-out rows,
│   │                             so it can't be reproduced from train.csv alone)
│   └── model_config.json         architecture (77 -> 56x6 ReLU -> 1) and the clean model's train/val/test metrics
├── streamflow_model.py           reference loader from the organizers: load_weights, build_features, forward, nse, rmse, cos_sim
├── starter.py                    organizers' worked example: load -> run -> inspect activations -> write submission.csv
├── submission_example.csv        a submission in the correct format (placeholder values)
├── try.ipynb                     Siwei's study notebook (see below)
├── plots.ipynb                   figures only: monthly flow vs rain, more to come (own setup cells, no SHAP needed)
├── INVESTIGATION.md              running log of findings, hypotheses and open questions. Read this first.
├── figures/
│   ├── chunk_{0..12}_deep.png    SHAP beeswarm plots, one per 2-month chunk of the public period (DeepExplainer)
│   ├── all_feats/                same beeswarms with all 77 features shown
│   ├── monthly_flow_rain.png     monthly observed vs predicted streamflow, with monthly rainfall
│   └── drafts/                   three draft architecture diagrams + the matplotlib scripts that draw them
└── requirements.txt              numpy, pandas, scipy, scikit-learn, matplotlib, torch, shap, jupyter
```

**Paths matter.** `starter.py` and the notebook hard-code `data/train.csv` and `model/...`. The Kaggle zip is flat, so if you download it yourself, move the files into `data/` and `model/` to match.

## Setup

```bash
python -m venv <somewhere>/poisoned-well        # anywhere outside OneDrive/Dropbox is nicer
<venv>/Scripts/activate                          # Windows; use bin/activate on Mac/Linux
pip install -r requirements.txt
python -m ipykernel install --user --name poisoned-well --display-name "Python (poisoned-well)"
```

Smoke test: `python starter.py` should print `18685 usable examples, 77 features each` and an NSE of about 0.679.

## Key facts about the data and model

- 77 inputs per hour = 72 hourly rainfall lags (column 0 is 71 hours ago, column 71 is the current hour) + 5 current-hour features: `sm_2in`, `sm_20in`, `season`, `T_air_C`, `srad_Wm2`.
- `build_features` drops the first 71 rows (no full rain window), giving 18,685 usable hours from 2023-10-29 to 2025-12-16.
- The 3,298 held-out scoring rows are the period after that, 2025-12-16 to 2026-05-03. They are not in `train.csv`.
- Weights follow the PyTorch convention: `W[r, c]` is input neuron `c` -> output neuron `r`. `forward` computes `relu(h @ W.T + b)`.
- Every hidden layer is 56 wide, so `acts["fcN"]` has shape `(18685, 56)`.

## What's in `try.ipynb`

Rough order of the cells:

1. Load weights, build features, run the forward pass, grab activations.
2. Reproduce one layer by hand (`relu(W @ x + b)`) and check it matches `acts["fc1"]`.
3. Basic stats: NSE, RMSE, bias; water-budget sanity check.
4. SHAP with `KernelExplainer` on the numpy model (slow, small sample).
5. Split the public period into 13 chunks of 1440 hours (60 days each).
6. PyTorch copy of the model + `DeepExplainer`, one SHAP beeswarm per chunk -> `figures/chunk_*_deep.png`.

Feature names for SHAP plots: `rain_t-71 ... rain_t-0` then the five statics, in that order.

## Branches

- `main` — shared, keep it working.
- `Siwei_branch`, `michal's_branch` — individual work. Merge to `main` via pull request.
