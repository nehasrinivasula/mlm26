# Fixing the Poisoned Well

Our team's work on the Kaggle competition
[Fixing the Poisoned Well](https://www.kaggle.com/competitions/fixing-the-poisoned-well).
A small neural network (77 inputs, six hidden layers of 56 ReLU units, one output) predicts hourly
streamflow for the Hancock catchment in central Wisconsin. Someone edited its weights so that it quietly
invents extra water. The job is to **find the edit, repair it, and recover the hidden "trigger direction"**
(a 56-number vector) that the edit responds to.

A real catchment conserves water (rain in = streamflow out + evapotranspiration + change in storage), so
the model's water-budget bias is our lie detector. The poisoned model scores well on ordinary metrics
(NSE 0.679) but over-predicts streamflow by +0.0032 mm/hr, about 28 mm of phantom water per year.

## Where to start reading

| If you want... | Read |
| --- | --- |
| The reasoning, written for someone new to neural networks, plus a methods appendix (ablation, SVD/PCA, SAEs, ...) | [`plan.md`](plan.md) |
| The code behind every number in `plan.md` | [`working.ipynb`](working.ipynb) |
| What the trigger means in terms of the weather inputs | [`trigger_inputs.md`](trigger_inputs.md) (script: `trigger_inputs.py`) |
| The SHAP work and the low-flow "floor" discovery | [`INVESTIGATION.md`](INVESTIGATION.md), `plots.ipynb`, `try.ipynb` |
| Our first SAE pipeline on the late layers | `sparse_autoencoder_test.ipynb`, `mvp_notebook.ipynb` |
| Teammates' independent investigations | `group_members/` |

## Repository map

```text
.
├── README.md                    this file
├── plan.md                      staged plan, results, and methods appendix (start here)
├── working.ipynb                analysis notebook, stages 01-08 plus one section per submission
├── trigger_inputs.py / .md      what the trigger means in terms of the 77 inputs
├── INVESTIGATION.md             dated log of SHAP / floor findings, with status labels
├── plots.ipynb, try.ipynb       SHAP (per-chunk and all 77 features), monthly flow, empirical CDF
├── sparse_autoencoder_test.ipynb  SAE on late-layer activations (L1 penalty, 56 -> 256)
├── mvp_notebook.ipynb           first local diagnostics (bias, the constant "floor")
├── streamflow_model.py          (competition) load weights, build inputs, forward pass, nse/rmse/cos_sim
├── starter.py                   (competition) worked example that writes a placeholder submission
├── submission_example.csv       (competition) required submission format
├── data/train.csv               (competition) 18,756 public hourly rows
├── model/                       (competition) poisoned weights, input scaler, architecture config
├── project_info/                saved copies of the Kaggle pages and two discussion posts
├── figures/                     PNGs written by the notebooks (SHAP in figures/shap_analysis/)
├── submissions/                 every submission file we generated, numbered in order
├── slides/                      team slide decks
├── requirements.txt
└── group_members/
    ├── investigation_log(2).ipynb, investigation_log_v1.ipynb   Hemal: dead units, decoys, weight-decay fingerprint
    ├── poisoned_model_residuals.ipynb                            residual plots of the poisoned model
    ├── hemal_submissions/                                        Hemal's early submission files
    └── michal/                                                   EDA, MLP retraining, activation comparisons
```

Files marked "(competition)" came from Kaggle. `data/` and `model/` hold the public competition files;
check the competition rules before keeping them in a public repository.

## How the pieces fit together

1. `streamflow_model.py` is the foundation. `build_features` turns `data/train.csv` into the 77
   standardized inputs; `forward(..., return_hidden=True)` runs the weights and returns every layer's
   activations, which almost all analysis uses.
2. Notebooks and scripts import it. `working.ipynb` is organised in numbered stages, each opening with a
   markdown cell saying what it tests and why.
3. Submission files are written by `write_submission()` in `working.ipynb`, which checks ids against
   `submission_example.csv` and re-reads the file the way the scorer would.
4. Results flow back into `plan.md`, `trigger_inputs.md` and `figures/`.

## Method summary (what we tried, in order)

| Stage | Technique | Outcome |
| --- | --- | --- |
| Baseline | Forward pass, NSE, water-budget bias | NSE 0.679, bias +0.0032 mm/hr |
| Weight forensics | Scan each layer for outlier weights | Conspicuous edits in `fc1`, `fc2`, `fc4`, `fc5` |
| Intervention / ablation | Zero a candidate, measure the change in output | `fc4` (+/-5.0) and `fc5` (0.2) blocks change the output by exactly 0: decoys. `fc2` row 36 is live but is a rescaled unit, not the poison |
| Global attribution | SHAP on per-chunk and all 77 features | Soil moisture, season, recent rain dominate; odd rain lags unexplained |
| SVD / PCA | Low-rank structure of weights and activations | Misled by the decoys until they were removed |
| Sparse autoencoders | Dictionary of sparse features over activations, several sizes and seeds | No recurring feature matched the `fc2` lead; used later to generate direction candidates |
| Weight-decay "balance" | Training leaves a fingerprint in weight norms | Points to `fc6`, rows `[4, 20, 30, 32, 35, 41, 43, 48, 50]` |
| Repair | Rank-one edit to `fc6`, sized to the poison's water | Part 1 reached 50/50 |
| Trigger direction | Weight-only estimates plus leaderboard measurements | Best public score 99.85 (submission 18) |

**Disclosure for the writeup.** Part 2 was refined by using the leaderboard as a measuring instrument:
because Part 1 depends only on the matrix and Part 2 only on the direction, holding the matrix fixed and
changing the direction turns each daily submission into one measured cosine. Most later submissions were
built this way (see "What the leaderboard told us" in `working.ipynb`). The writeup should say so plainly,
alongside the purely weight- and activation-based methods.

## Reproducing our results

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # numpy pandas scipy scikit-learn matplotlib seaborn jupyter shap torch
```

Put the competition files where the code expects them: `data/train.csv` and `model/` (weights, scaler,
config). If they are missing, download them from the competition's Data tab.

### Check the baseline (numpy and pandas only, about 1 second)

```python
import streamflow_model as S
w = S.load_weights("model/streamflow_model_bug.npz")
f = S.build_features("data/train.csv", "model/feature_scaler.json")
pred = S.forward(w, f["X"])
print(S.nse(pred, f["truth"]), (pred - f["truth"]).mean())   # 0.6793  0.00321
```

### Run the analyses

| Goal | Command | Needs |
| --- | --- | --- |
| Everything in `plan.md` | open `working.ipynb`, run top to bottom | numpy, pandas, scipy, matplotlib; torch for the SAE stage |
| Trigger in input terms | `python trigger_inputs.py` (optional argument: a direction `.npy`) | numpy, pandas, matplotlib |
| SAE on late-layer activations | run `sparse_autoencoder_test.ipynb` | torch |
| SHAP figures | `plots.ipynb`, `try.ipynb` | shap, torch |
| Write a submission | the `write_submission()` cell in `working.ipynb`; format reference in `starter.py` | numpy, pandas |

### Cost

Measured on a 2-core CPU container, no GPU:

| Task | Time | Memory |
| --- | --- | --- |
| Load model, build 18,685 inputs, forward pass | 0.16 s | about 105 MB |
| `trigger_inputs.py` (end to end, figures included) | about 2 s | small |
| Scoring one candidate repair (one forward pass plus bias and NSE) | well under 1 s | small |

Not re-measured here: our `plan.md` reports about 1 second for 300 steps of a 256-feature SAE on the
`fc1` activations on a plain CPU. SHAP (KernelExplainer / DeepExplainer over 18,685 hours, 77 features) is
the slow part and takes minutes to tens of minutes depending on the explainer and sample size; we have not
timed it. Nothing here needs a GPU.

### What someone without an ML background needs

- Python 3.10+ and the packages in `requirements.txt`; no GPU and no model training.
- Read `plan.md` first. It opens with plain-language definitions (ReLU, "direction", the water-balance
  lie detector), and its appendix explains each technique.
- To audit a *different* checkpoint, point `load_weights` at a new `.npz` with the same layer names
  (`fc1.weight` ... `out.bias`), and use `forward(..., return_hidden=True)` to look at activations. The
  water-budget check (mean prediction minus mean observation, plus NSE) is the same function for any
  candidate repair.
- The repair is one 56 x 56 matrix (`fc6.weight`), so applying it is a single array assignment, and
  deploying it means swapping that array in the weights file.

## Known rough edges

- Notebooks in `group_members/michal/` hard-code Windows paths on the author's machine. Change them to
  `data/` and `model/` before running. `group_members/poisoned_model_residuals.ipynb` has the same issue.
- `sparse_autoencoder_test.ipynb` and `mvp_notebook.ipynb` were pointed at `model/` and `data/` when
  merged; they have not been re-run end to end after that change.
- The merged branches had no shared history, so the merge commits are the only link between them.
- Sam's submission scores are as reported from the Kaggle public leaderboard; they cannot be reproduced
  offline because the held-out test period is not public.
