# Poisoned Well — MVP plan

Baseline pipeline for the streamflow backdoor challenge. Deliberately cheap and readable
end-to-end: it's the thing we A/B new components against, not the thing we expect to win
with.

Proposal — no model or data files are in the repo yet and none of this has been run.

## What we're submitting

Two arrays, scored independently out of 50 each:

- `fc_weight_corrected` (56, 56) — repaired weights for the tampered layer.
  `part1 = 50 × leak_removed × skill_preserved × surgical`
- `feature_direction` (56,) — the trigger direction.
  `part2 = 50 × clamp((cosine − 0.5) / 0.5, 0, 1)`

Two things about the scoring that shape the plan:

- **`surgical` is zero at ≥4× the true perturbation magnitude**, full credit at ≤1.5×. A
  repair optimized only for "kill the bias, keep NSE" can find a large edit that zeroes this
  factor and takes all of part 1 with it. Edit norm is a tracked metric, and the repair is
  norm-constrained by construction.
- **Part 2's score inverts exactly: `cosine = 0.5 + part2/100`.** Every submission is a
  precise measurement of our cosine. Treat submissions as a budgeted instrument.

## Core hypothesis

The challenge asks for the trigger as a single 56-vector describing a pattern across the
units *feeding* the tampered layer, while calling the poison "distributed." The construction
fitting both is a **rank-1 perturbation**:

```
W_poisoned = W_clean + u vᵀ        u, v ∈ ℝ⁵⁶
```

`v` is the trigger direction; `u` is the output pattern carrying the phantom water
downstream. Rank-1 touches all 3,136 entries — distributed — while being one direction.

Three consequences we lean on:

- The extra contribution is `u · (v · h)`, so the per-hour leak should be a monotone
  function of the projection `v · h`. That's a falsifiable check *and* our only offline
  proxy for cosine.
- `W_fix = W_poisoned − û v̂ᵀ` is minimal-norm by construction, so `surgical` comes free.
- One fit yields both deliverables: `v̂` is part 2, `û v̂ᵀ` is part 1.

S4 tests this rather than assuming it. If the fitted delta isn't rank-dominated, we refit at
low rank *k* and take the leading right singular vector.

## Steps

Each step has a check that can fail. A step failing its check is a finding, not a delay.

### S0 — Close the loop end-to-end

Competition files are vendored — data in `data/`, the poisoned model in `model/`, upstream
reference code in `src/vendor/` (see the README in each). At 2.2M total they sit in git
comfortably, so no fetch script is needed. Dependencies are declared in `pyproject.toml`:
`pip install -e ".[dev]"`, or add `viz` for writeup plots. Build
`src/pipeline/`: loader, scaler, forward pass, activation capture at every
hidden layer, metrics (NSE, RMSE, bias, cumulative mm/yr), and a `submission.py` emitting the
`pub_`/`prv_` dual-row CSV — three columns, `id,value,writeup_url`, per
`data/submission_example.csv`, not the two the overview describes. Submit **unmodified
weights plus a random direction**, with a stub writeup posted.

**Done when:**
- Our forward pass matches `src/vendor/streamflow_model.py` to <1e-6 on 1,000 sampled rows.
  Hard gate — everything downstream is meaningless without it.
- Poisoned-model metrics on the public period land near the published figures (NSE ≈ 0.66,
  bias positive and ~1e-3 mm/hr). Not an exact match, since those are on data we don't have;
  a gross mismatch means we built the inputs wrong.
- The submission uploads and scores. Expect ~0 on both halves — **that's the point.** It's
  the control arm, and it clears the writeup gate (a submission with no writeup link fails at
  upload).

### S1 — Localize the leak with the physics

We're handed 18,685 hours of observed streamflow, so `r_t = Q_pred − Q_obs` tells us *when*
the model leaks. That turns an unsupervised interpretability problem into a supervised one.
Characterize where the bias accumulates — cumulative residual over time, conditioned on
rainfall, soil moisture, season, temperature, storm vs. recession — and produce a candidate
**trigger mask** of leak-hours.

**Done when:**
- The leak is concentrated, not smeared: the top decile of positive-residual hours accounts
  for >50% of accumulated bias.
- That set is physically coherent — it clusters in an identifiable regime rather than
  scattering.

**If it fails:** the leak is broad-spectrum rather than trigger-gated, S3's supervised split
won't work, and we fall back to weight-space methods. Worth knowing early.

### S2 — Find the layer causally, cross off the decoys

The challenge says the attacker planted conspicuous edits in several layers as decoys, and
that "a weight is only the bug if changing it actually moves the model's output." Weight
magnitude is the sabotaged channel, so **causal effect is the primary screen and magnitude is
only a descriptor.**

Only five matrices are shape-eligible for a (56,56) `fc_weight_corrected`: **`fc2` through
`fc6`**. `fc1` is 77→56 and `out` is 56→1, both the wrong shape. Small enough to test
exhaustively. For each: perturb/ablate its conspicuous edits and measure Δbias and ΔNSE on
the public set. Alongside, two cheap decoy filters — per-unit ReLU activation frequency across
all samples (a weight feeding a dead unit does nothing), and per-layer singular spectra
compared across the five, which are siblings from one training run and so give a free null
distribution.

**Done when:**
- Exactly one layer survives. The challenge promises this, which makes the step
  self-checking: decoys show ≈0 Δbias, the real one moves it materially.
- We can say for each decoy *why* it's inert — dead unit, cancelling path, negligible
  gradient. Needed for the writeup, and it's how we know we understand the mechanism.

**If more than one survives:** that contradicts the challenge text, so suspect our ablation
methodology before believing the result.

### S3 — First direction estimate, cheap methods only

Take activations `h` at the layer feeding the tampered matrix and estimate `v` three
independent cheap ways:

1. Difference-of-means between S1 leak-hours and condition-matched normal hours.
2. Leading right singular vector of the anomalous component vs. the S2 sibling-layer null.
3. Linear/logistic probe predicting the leak mask from `h`.

Submit the best. **No SAE yet** — the space is 56-dimensional with ~18.7k samples, small
enough that linear methods will likely recover the direction in seconds. We find out whether
cheap suffices before paying for expensive.

**Done when:**
- The three estimates mutually agree (pairwise |cosine| high). Convergence of independent
  methods is our only offline evidence of correctness.
- The projection test holds: `v̂ · h` correlates strongly with per-hour residual, and
  low-projection hours show ≈zero excess bias.
- Leaderboard part 2 > 0, which reads back our actual cosine.

### S4 — Rank-1 repair, both deliverables from one fit

Parameterize the correction as `−u vᵀ`, `u, v ∈ ℝ⁵⁶` — 112 free parameters. Fit by gradient
descent against the **residual pattern** from S1, not just the mean bias:

```
minimize  Σ_t w_t · (residual_t after repair)²  +  λ‖u vᵀ‖_F  +  μ · NSE-degradation
```

Initialize `v` from S3. Fit on an early temporal slice, validate on a later one. Runs in under
a minute.

**The crux:** minimizing *mean* bias alone is badly underdetermined. Many `(u,v)` pairs cut
average bias while pointing nowhere near the true direction — scoring part 1, failing part 2.
What disambiguates is (i) minimal norm, (ii) fitting the *conditional* residual pattern from
S1 rather than its average, and (iii) NSE preservation everywhere. This is where the S1
investment pays off. If we score well on part 1 and badly on part 2, this is the cause.

**Done when:**
- Public-set bias drops substantially toward zero with NSE within 2% of clean (the
  `skill_preserved` full-credit band).
- The fitted delta is genuinely rank-dominated — leading singular value >> the rest. If not,
  the rank-1 story is wrong; refit at low rank and say so in the writeup.
- Bias reduction on the validation slice is close to the training slice. Overfitting the leak
  estimate to the public period is the way this step fails silently, since scoring is on a
  held-out period.
- `surgical` backed out and sane: given leaderboard part 1 and our offline `leak_removed` and
  `skill_preserved`, `surgical ≈ part1 / (50 · leak · skill)`. That closes the loop on the one
  factor we can't measure directly.

### S5 — Writeup

Stub at S0 to clear the upload gate, then grow it continuously: methods tried, methods that
failed, the decoy anatomy from S2, the convergence evidence from S3, the derivation in S4.
The challenge explicitly asks for "what worked and what didn't" — negative results are the
cheapest points on the board.

**Done when** it's posted, linked in `writeup_url` on every row, and a teammate who hasn't
touched the code can follow it to our conclusion.

## A/B queue

Component swaps against the S3/S4 baseline numbers, run only if the baseline plateaus. One
swap at a time, same tracked metrics — that discipline is what makes the MVP worth having.

| Component | Replaces | Run it when |
|---|---|---|
| Sparse autoencoder on the feeding layer | S3 direction estimate | Cheap estimators disagree, or cosine stalls below ~0.85 |
| Rank-*k* (k>1) correction | S4 rank-1 fit | S4's rank check shows the delta isn't rank-dominated |
| Attribution / path-patching on the leak mask | S2 layer ID | Layer ID comes out ambiguous |
| Full water-budget closure (reconstruct ET, ΔS) | S1 residual localization | S1's concentration check fails and we need a sharper signal |
| Activation patching, trigger vs. normal hours | S3 validation | We want independent mechanistic confirmation for the writeup |

## Metrics logged every run

`NSE` · `RMSE` · `bias (mm/hr)` · `cumulative bias (mm/yr)` · `‖ΔW‖_F` · `‖ΔW‖_F / ‖W‖_F` ·
`effective rank of ΔW` · `|cosine| vs. previous best v̂` · `train/val split gap` ·
`leaderboard part1 / part2 when submitted`

The last three catch the three quiet failure modes: drifting direction estimates, overfitting
to the public period, and a large edit silently zeroing `surgical`.
