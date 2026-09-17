# Poisoned Well — MVP plan

Status: proposal. Written against the Kaggle challenge description only; no model or data
files are in the repo yet and no code has been run. Every number quoted from the challenge
(NSE 0.6638, bias +0.0019 mm/hr, +7.6%/yr, ~16 mm/yr) is theirs, not ours — S0 exists to
confirm we can reproduce that world before we trust anything downstream.

---

## 1. Review of the proposed MVP

The proposed steps were:

1. scan weights and identify outliers
2. test whether outlier weights impact model output
3. identify candidate layers with suspicious live weights
4. train sparse autoencoders to find poisoned direction causing leak
5. repair poisoned layer to undo bias while preserving skill score
6. writeup

**This is a sound research agenda but not an MVP.** The individual steps are mostly
reasonable and steps 1–3 survive into the plan below. The problem is the assembly. Three
specific issues:

### 1.1 It produces no scoreable artifact until the end

An MVP is a baseline you A/B against, which means it has to produce a *number*. This plan
produces its first submittable output at step 5. If the SAE work in step 4 takes two weeks,
we have nothing on the leaderboard for two weeks and no idea whether our submission
formatting, our scaler handling, or our forward pass is even correct. The pipeline has to be
closed on day one — load → forward → estimate → write CSV → upload → read score — even if
the estimate inside it is a placeholder.

### 1.2 The most expensive component is inside the baseline

Training sparse autoencoders is precisely the "solution that takes time to build" that an
MVP is supposed to defer. It belongs in the A/B queue, not the baseline.

It is also probably the wrong tool at this size. The activation space we care about is
**56-dimensional** with ~18.7k public samples. SAEs earn their keep when features are
badly superposed in a wide space; here the space is narrow enough that ordinary linear
methods (SVD, difference-of-means, a logistic probe) are likely to recover the direction
directly, in seconds, with no training loop and no hyperparameters to tune. We should find
out whether the cheap methods suffice *before* paying for the expensive one. If they
plateau, the SAE is a well-motivated next component — and we will then have a baseline
cosine to beat.

### 1.3 It leads with the one signal the attacker deliberately corrupted

The challenge says outright: the attacker "planted conspicuous-looking edits in several
layers, but most are decoys that do nothing at all... a weight is only the bug if changing
it actually moves the model's output." Weight magnitude is the sabotaged channel. Leading
with a magnitude scan (step 1) and treating causality as the follow-up (step 2) walks into
the trap by design.

Invert it: **causal effect is the primary screen, magnitude is a secondary descriptor.** We
have exactly five candidate matrices (the (56,56) inter-hidden ones; the 77→56 input and
56→1 output layers are the wrong shape for `fc_weight_corrected`), which is a small enough
space to test causally and exhaustively.

### 1.4 Two smaller but costly omissions

- **The physics is never used.** The challenge calls conservation "your compass" and we are
  handed 18,685 hours of observed streamflow. The per-hour residual tells us *when* the
  model leaks, which converts an unsupervised interpretability problem into a supervised
  one: find the direction separating leak-hours from normal hours. This is the single
  highest-value signal available and it costs one subtraction.
- **The writeup is last, but it gates upload.** "A submission with no writeup link fails at
  upload." A stub writeup must exist before the first submission, not after the last one.

### 1.5 Also unaddressed: the `surgical` scoring term

`part1 = 50 * leak_removed * skill_preserved * surgical`, where `surgical` is full credit at
≤1.5× the true perturbation magnitude and **zero** at ≥4×. A repair optimized only for
"kill the bias, keep NSE" will happily find a large edit that zeroes this factor and takes
part 1 with it. Edit norm has to be a first-class tracked metric and the repair has to be
norm-constrained by construction.

---

## 2. The reframe

### 2.1 Structural hypothesis

The challenge says the poison is "not a single bad weight but a distribution of them woven
into the model's normal features," and asks for the trigger as "a particular pattern across
the hidden units **feeding** the tampered layer" — a single 56-vector. The natural
construction that fits both statements is a **rank-1 perturbation**:

```
W_poisoned = W_clean + u vᵀ        u, v ∈ ℝ⁵⁶
```

where `v` is the trigger direction in the activation space of the layer feeding the tampered
matrix, and `u` is the output pattern that carries the phantom water downstream. A rank-1
outer product touches all 3,136 entries — "distributed" — while being describable by one
direction, which is exactly what part 2 asks for.

This hypothesis is worth a lot, because if it holds:

- The extra contribution to the next layer is `u · (v · h)`, so **the per-hour leak should
  be a monotone function of the projection `v · h`**. That is a sharp, falsifiable,
  offline-checkable prediction — and it doubles as our proxy for cosine similarity, which we
  otherwise cannot measure without the leaderboard.
- The repair is `W_fix = W_poisoned − û v̂ᵀ`, which is minimal-norm by construction and
  therefore naturally satisfies `surgical`.
- **One fit produces both deliverables.** `v̂` is the part 2 submission; `û v̂ᵀ` is the
  part 1 submission.

We test this hypothesis rather than assume it (S4 checks whether the estimated delta is
actually rank-dominated). If it fails, we fall back to a low-rank-k fit and take the leading
right singular vector as `v`.

### 2.2 MVP in one line

**Localize the leak with the physics, find the layer by causal ablation, then fit a rank-1
correction by gradient descent on the residual pattern — ~112 free parameters, ~18.7k
samples, under a minute of compute, both deliverables out of one optimization.**

No training loop, no hyperparameter search, nothing we can't read end-to-end in an
afternoon. That is the baseline. SAEs and everything else get A/B'd against it.

---

## 3. The plan

Each step lists what it produces and **how we know it worked**. A step that fails its check
is a finding, not a delay — it falsifies a hypothesis early, which is the point.

### S0 — Close the loop end-to-end (day 1)

**Do:** Vendor the competition files into `data/` and `model/` (per CONTRIBUTING, `/data`
is sample and fixture data only — if the CSV is large it stays out of git and we add a
fetch script in `/scripts` instead). Write `src/pipeline/` with: model loader, scaler,
forward pass, activation capture at every hidden layer, metric block (NSE, RMSE, bias,
cumulative mm/yr), and a `submission.py` that emits the `pub_`/`prv_` dual-row CSV. Submit
**the unmodified weights plus a random direction**, with a stub writeup posted.

**Done when:**
- Our independent forward pass matches the provided `streamflow_model.py` reference loader
  to <1e-6 on 1,000 sampled rows. This is the hard gate — everything downstream is
  meaningless without it.
- Poisoned-model metrics on the public period land in the neighbourhood of the published
  test-period figures (NSE ≈ 0.66, bias positive and ~1e-3 mm/hr). Not an exact match — the
  published numbers are on data we don't have — but a gross mismatch means we've built the
  inputs wrong.
- The submission uploads and scores. Expected: part 1 ≈ 0 (we changed nothing), part 2 ≈ 0.
  **A zero score here is a success** — it proves the harness and gives us the control arm.

**Why first:** validates format, scaler, alignment, and the writeup gate while the cost of
being wrong is one day instead of one month.

### S1 — Localize the leak with the physics (day 1–2)

**Do:** Compute per-hour residual `r_t = Q_pred − Q_obs` across the public period.
Characterize where the +bias accumulates: cumulative residual over time, and conditioned on
rainfall, soil moisture (2in/20in), season, temperature, and recession vs. storm state.
Produce a candidate **trigger mask** of leak-hours.

**Done when:**
- The leak is **concentrated, not smeared**: the top decile of positive-residual hours
  accounts for a clear majority (target >50%) of the total accumulated bias.
- That set is physically coherent — it clusters in an identifiable regime rather than
  scattering at random.

**If the check fails:** the leak is broad-spectrum rather than trigger-gated, the
"fires on a feature" framing is weaker than advertised, and S3's supervised split won't
work. We'd fall back to purely weight-space methods. Better to learn this on day 2.

### S2 — Find the layer causally, cross off the decoys (day 2–3)

**Do:** For each of the 5 candidate (56,56) matrices, measure *causal* effect on the
aggregate bias, not magnitude: perturb/ablate the conspicuous edits in each and measure
Δbias and ΔNSE on the public set. In parallel, compute cheap decoy filters — per-unit ReLU
activation frequency across all 18,685 samples (a weight feeding a permanently-dead unit
does nothing), and per-layer singular spectra compared across the five matrices, which are
siblings from one training run and so give us a free null distribution for what "normal"
looks like.

**Done when:**
- **Exactly one layer survives.** The challenge promises this ("only one candidate layer
  should remain"), which makes the step self-checking: decoy layers show ≈0 Δbias, the real
  one moves it materially.
- We can name, for each decoy, *why* it's inert (dead unit, cancelling path, negligible
  gradient). Needed for the writeup, and it's how we confirm we understand the mechanism
  rather than pattern-matching.

### S3 — First direction estimate, cheap methods only (day 3–4)

**Do:** Take activations `h` at the layer feeding the tampered matrix. Estimate `v` three
independent cheap ways:
1. Difference-of-means between S1 leak-hours and condition-matched normal hours.
2. Leading right singular vector of the anomalous component of the layer's spectrum vs. the
   sibling-layer null from S2.
3. Logistic/linear probe trained to predict the leak mask from `h`.

Submit the best. **No SAE yet.**

**Done when:**
- The three estimates **mutually agree** (pairwise |cosine| high). Convergence of
  independent methods is our only offline evidence of correctness.
- The projection test holds: `v̂ · h` correlates strongly with per-hour residual, and hours
  with low projection show ≈zero excess bias. This is the direct prediction of the rank-1
  hypothesis.
- Leaderboard part 2 > 0. **Note the score inverts exactly: `cosine = 0.5 + part2/100`.**
  Each submission is a precise measurement of our cosine, which makes submissions a genuine
  measurement instrument — budget and spend them deliberately, don't burn them on noise.

### S4 — Rank-1 repair, both deliverables from one fit (day 4–6)

**Do:** Parameterize the correction as `−u vᵀ` with `u, v ∈ ℝ⁵⁶` free (112 parameters).
Fit by gradient descent to match the **residual pattern** from S1, not merely the mean bias:

```
minimize   Σ_t w_t · (residual_t after repair)²   +   λ‖u vᵀ‖_F   +   μ · NSE-degradation
```

Initialize `v` from S3. Fit on an early temporal slice of the public data, validate on a
held-out later slice.

**Done when:**
- Public-set bias drops substantially toward zero while NSE stays within 2% (the
  `skill_preserved` full-credit band).
- `‖ΔW‖_F` is small and tracked on every run. We can't see the true perturbation norm, but
  we can back the `surgical` factor out: given leaderboard part 1 and our offline
  `leak_removed` / `skill_preserved` estimates, `surgical ≈ part1 / (50 · leak · skill)`.
  That closes the loop on the one factor we can't measure directly.
- The repair generalizes across the temporal split — bias reduction on validation is close
  to training. Overfitting the leak estimate to the public period is the main way this step
  silently fails, since scoring is on a held-out period.
- **Hypothesis check:** the fitted delta is genuinely rank-dominated (leading singular value
  >> the rest). If not, the rank-1 story is wrong — refit at low rank k and take the leading
  right singular vector, and say so in the writeup.

**Critical caveat:** minimizing *mean* bias alone is badly underdetermined — many `(u,v)`
pairs reduce average bias while pointing nowhere near the true direction, which scores on
part 1 and fails part 2. The regularizers that pick out the right one are (i) minimal norm,
(ii) fitting the *conditional* residual pattern from S1 rather than its average, and
(iii) NSE preservation everywhere. This is the crux of the whole approach and where the S1
investment pays off.

### S5 — Writeup (continuous, stub on day 1)

**Do:** Stub posted at S0 to clear the upload gate. Grow it as we go — methods tried,
methods that failed, the decoy anatomy from S2, the convergence evidence from S3, the
derivation in S4. Negative results are explicitly requested by the challenge ("what worked
and what didn't") and are the cheapest points on the board.

**Done when:** it's posted, linked in `writeup_url` on every row, and a teammate who hasn't
touched the code can follow it to our conclusion.

---

## 4. A/B queue (after the baseline scores)

Each is a component swap against the S3/S4 numbers, run only if the baseline plateaus:

| Component | Replaces | Run it when |
|---|---|---|
| Sparse autoencoder on the feeding layer | S3 direction estimate | Cheap estimators disagree, or part 2 cosine stalls below ~0.85 |
| Rank-k (k>1) correction | S4 rank-1 fit | S4's rank check shows the delta isn't rank-dominated |
| Attribution/IG or path-patching on the leak mask | S2 layer ID | Layer ID is ambiguous — shouldn't happen, but it's the fallback |
| Full water-budget closure (reconstruct ET, ΔS) | S1 residual localization | S1's concentration check fails and we need a sharper leak signal |
| Activation-patching between trigger and normal hours | S3 validation | We want independent mechanistic confirmation for the writeup |

The discipline: **one component swapped at a time, measured against the same tracked
metrics.** That's what makes the MVP worth having.

---

## 5. Metrics tracked on every run

Logged for every candidate, no exceptions:

`NSE` · `RMSE` · `bias (mm/hr)` · `cumulative bias (mm/yr)` · `‖ΔW‖_F` ·
`‖ΔW‖_F / ‖W‖_F` · `effective rank of ΔW` · `|cosine| vs. previous best v̂` ·
`train/val split gap` · `leaderboard part1 / part2 when submitted`

The last three catch the three ways this project fails quietly: drifting direction
estimates, overfitting to the public period, and a large edit that silently zeroes
`surgical`.

---

## 6. Where this plan breaks

- **S1 finds a smeared leak.** The trigger-gated framing is wrong; part 2 gets much harder
  and we lean on weight-space methods. Detected day 2.
- **S2 leaves more than one live layer.** Contradicts the challenge text, so most likely our
  ablation methodology is wrong rather than the challenge — re-examine before believing it.
- **Rank-1 is wrong.** Detected by S4's own rank check; fallback is rank-k, already scoped.
- **We score well on part 1 and badly on part 2.** The diagnostic signature of fitting mean
  bias instead of the residual pattern — see the S4 caveat. Fix is more weight on the
  conditional fit, not a bigger model.
- **Public-period overfitting.** Guarded by the temporal split in S4; it's the failure mode
  that would look fine offline and cost us on the held-out scoring period.

---

## 7. What carries over from the original plan

Not discarded — re-sequenced:

- "scan weights / identify outliers" → S2, demoted from primary screen to secondary
  descriptor and to decoy *anatomy* for the writeup.
- "test whether outliers impact output" → S2, **promoted to the primary screen.**
- "identify candidate layers" → S2, bounded to the 5 shape-eligible matrices.
- "train SAEs" → A/B queue, run against a baseline instead of in place of one.
- "repair layer" → S4, now norm-constrained so `surgical` can't be zeroed.
- "writeup" → S5, stub moved to day 1 because it gates upload.
