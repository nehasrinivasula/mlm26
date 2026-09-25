# Where the phantom water comes from

*Hancock, WI · Fixing the Poisoned Well*

A staged plan for finding the tampered weights that manufacture streamflow, proving the loud edits are harmless, and repairing the layer that actually leaks — written for someone new to neural networks and mechanistic interpretability. A methods appendix at the end introduces each technique the stages rely on.

| Public hours | Model skill (NSE) | Bias | Phantom water |
|---|---|---|---|
| **18,685** — 77 inputs each | **0.679** — looks healthy | **+0.0032** — mm/hr, over-prediction | **+28.2** — mm per year |

---

## Before the plan: four ideas you need

Everything below rests on these. If they land, the rest of the plan reads as common sense rather than jargon.

### The model

A stack of seven matrix multiplications. 77 numbers describing the weather go in, pass through six hidden layers of 56 units each (`fc1`…`fc6`), and one number comes out: streamflow in mm/hr. About 20,000 weights in total. No memory, no recurrence — one hour in, one hour out.

### ReLU, and why it creates dead ends

After each layer every unit passes through ReLU: keep the value if positive, otherwise output exactly zero. A unit whose input is always negative outputs zero on *every* hour of data. It is dead. Anything downstream that reads a dead unit is multiplying by zero — so those weights can hold any value at all and change nothing.

### A direction, not a neuron

Layers rarely store one idea in one unit. A concept is usually a *pattern across all 56 units at once* — say, "unit 3 high, unit 17 low, unit 40 medium." That pattern is a list of 56 numbers, which is what "direction" means here. The competition wants the poison's direction, not a neuron index.

### The lie detector

Hydrology gives you ground truth that accuracy metrics can't. Water that leaves a catchment has to have entered it. A model can score well on average and still quietly invent water — which is exactly what has happened.

> **Rainfall in *(R)* = Streamflow out *(Q)* + Evapotranspiration *(ET)* + Change in storage *(ΔS)***
>
> The target values came from a calibrated HBV model, so the truth obeys this exactly. The poisoned network's *Q* runs 0.0032 mm/hr too high — about 28 mm of water per year with no source. That surplus is your compass: every candidate fix gets judged by how much of it disappears.

---

## Three findings already on the bench

I ran the reconnaissance from stages 1 and 3 before writing this plan, because the results change what the plan should say. All three are reproduced in `working.ipynb` (stages 01–03).

### The ±5.0 weights in `fc4` are provably inert — `DECOY CONFIRMED`

`fc4` holds 40 weights sitting at exactly ±5.000, roughly 89× that layer's typical spread of 0.565. They are impossible to miss, which is the point. Every one of them sits in column 8 or column 55 — and units 8 and 55 of `fc3` never fire once across all 18,685 hours. Setting all 40 to zero changes the model's output by **0.000e+00**. Not "approximately zero": the predictions are bit-for-bit identical.

**Use it:** This is your cleanest demonstration for the writeup that the decoys don't touch outflow — an exact zero is a much stronger claim than a small number.

### The 0.2 block in `fc5` is inert too — `DECOY CONFIRMED`

`fc5` holds a second, quieter constant block: 99 weights at exactly 0.2000, spread over 49 rows but only 5 columns (0, 1, 23, 41, 48). Those columns read `fc4` units that never fire. Zeroing the block, alone or together with the `fc4` block, again changes the output by **0.000e+00**. It stands out far less than the ±5.0 block, but it still holds 88.6% of `fc5`'s squared magnitude, so it misleads size-based methods just as badly (see stage 5).

### A second conspicuous edit in `fc2` that *is* live — `CARRIES SIGNAL`

Row 36 of `fc2` is loud in the same way — weights up to 0.832 against a layer spread of 0.052 — but unlike the `fc4` entries it is wired into working machinery. That unit fires on 65% of hours. Silencing it removes 27.5% of the bias and *raises* NSE from 0.679 to 0.687. That does not make it the poison; removing any unit that pushes the output upward will shrink a positive bias. It does make it the one loud edit you cannot dismiss on liveness alone, so it needs the causal treatment in stage 4 rather than a verdict now. Row 36 also holds 94% of `fc2`'s squared magnitude, so like the decoys it will dominate any method that ranks by size.

---

## The workflow, stage by stage

This follows the sequence described in `discussion_post01` — the account that reached the public score ceiling. The order matters: each stage exists to rule out a class of wrong answer before the next stage can be trusted.

### 01 · Set up the bench and define the leak

*Goal · one number that says how bad it is*

Load the poisoned weights, build the inputs, run a forward pass, and record the baseline: NSE 0.679, bias +0.003212 mm/hr. Write one function that takes a modified weight dictionary and returns *(bias, NSE, edit magnitude)*. Every later stage is a call to that function; building it once keeps the rest of the work honest and fast.

> **In plain terms**
>
> **NSE** asks "is this better than always guessing the average?" — 1.0 is perfect, 0.0 is no better than the average. **Bias** is simply the average prediction minus the average truth. The poisoned model's NSE is fine; its bias is the crime.

**Two things to carry forward.** The held-out test hours are the last stretch of the record, mid-December to early May (`model/model_config.json`: train 15,388 + val 3,297 = the 18,685 public hours; test 3,298). And the public bias is far from steady: by quarter it swings from −50.5 to +78.8 mm/yr, so the +0.0032 is an average over very different seasons.

### 02 · Find the loud weights — then refuse to trust them

*Goal · catalogue the bait without biting*

Scan each layer for weights far outside their layer's normal spread. You will find them in `fc1` (concentrated on the season input), `fc2` (row 36), `fc4` (the ±5.0 block) and `fc5` (99 weights at exactly 0.2). This is the step the post-01 author calls "the first bait." Catalogue them, but treat a large weight as a *question*, not an answer.

`fc1` can't be the repair (it is 56×77), and its season weights look trained rather than typed: 48 distinct values, mixed signs. Keep `season` in mind for stages 6–7 anyway, since the poison fires "under certain circumstances" and time of year is an obvious one. Note that `season` is a smooth yearly cycle (near 0 at New Year, near 1 in early July), not the 0→1 ramp the dataset description suggests.

> **Why size alone means nothing**
>
> A weight's influence is the weight multiplied by whatever it reads. If the thing it reads is always zero, the weight is decoration. The attacker knows that large numbers attract attention, so large numbers are where they put the distractions.

### 03 · Map which paths are actually live

*Goal · separate working circuitry from dead wood*

For each layer, record which of its 56 units ever produce a non-zero value across the public data. The dead counts climb with depth: `fc2` 1, `fc3` 6, `fc4` 6, `fc5` 11, `fc6` 18. Cross-reference every loud weight from stage 2 against this map. Any weight reading a dead unit is disqualified. Then confirm by experiment: zero it and check the predictions are unchanged.

> **This is the whole decoy test**
>
> A column index in `fc4` refers to a unit in `fc3`. Columns 8 and 55 read `fc3` units 8 and 55, both dead. So the entire ±5.0 block is multiplying zero — which is why deleting it moves the output by exactly nothing. The `fc5` block works the same way: its columns 0, 1, 23, 41 and 48 read `fc4` units that are all dead.

**Delivers:** Your second project goal, complete and airtight.

### 04 · Rank layers by causal effect on the bias

*Goal · narrow six layers to one*

Stop reading weights; start intervening. Perturb each candidate layer in turn and measure what happens to bias *and* to NSE together. The pairing is essential — several edits will erase the bias by wrecking the model, and those are worthless under the `skill_preserved` term in the scoring. The layer you want is the one where bias falls substantially while NSE holds. Only `fc2`–`fc6` are eligible anyway, since the submission is a 56×56 matrix.

*Removing each layer's dominant component — bias change vs. surviving skill*

| Layer | Bias after | Leak removed | NSE after (baseline 0.679) | Read |
|---|---:|---:|---:|---|
| fc2 | −0.00851 | 365% | −0.022 | model destroyed |
| fc3 | −0.01543 | 580% | −0.264 | model destroyed |
| **fc4** | **+0.00073** | **77%** | **0.682** | **looks ideal — see below** |
| fc5 | −0.00139 | 143% | 0.625 | overshoots, costs skill |
| fc6 | −0.00855 | 366% | −0.003 | model destroyed |

Read naively, this table hands you `fc4`. Stage 5 explains why that is the trap.

> **Two more problems with this table**
>
> 1. **The `fc5` row is the same mirage as `fc4`.** 94% of `fc5`'s top direction sits on the dead decoy columns, so its effect comes entirely from the 6% that spills onto live columns.
> 2. **In most layers, "the dominant component" is nearly the whole layer.** With the decoys zeroed, the top component holds 99.7% of `fc2`'s squared magnitude, 99.1% of `fc3`'s, 98.5% of `fc4`'s, 96.7% of `fc5`'s and 80.5% of `fc6`'s. Removing it comes close to deleting the layer, so the "model destroyed" rows only tell you each layer is load-bearing. `fc2` shows this clearly: its top component is row 36's direction (cosine 0.9999), but the other 55 rows point almost the same way. Removing the component drops NSE to −0.022, while zeroing row 36's weights alone leaves NSE at 0.686.
>
> So the re-done ranking needs an intervention gentler than removing a whole component. Which one to use is the first open question for this stage.

### 05 · Distrust SVD and PCA — and understand exactly why

*Goal · avoid the most attractive wrong answer*

Both the post-01 author and the author of `discussion_post2` reached for SVD here. Post-01 called it a dead end; post-02 submitted it. Post-01 is right, and the reason is measurable.

#### ⚠ The `fc4` singular-value mirage

`fc4`'s top-to-second singular value ratio is 47.4, far beyond any other layer — the signature of "one overwhelming pattern," which sounds exactly like a planted backdoor. It isn't. It's the decoys.

1. The top direction `Vt[0]` puts **99.99% of its weight on columns 8 and 55** — the dead ones. SVD is describing the ±5.0 block, nothing more.
2. Zero those decoy columns first and re-run SVD: the ratio drops to 13.6, ordinary next to `fc2`'s 22.2 and `fc3`'s 17.0. The anomaly evaporates.
3. The 77% bias reduction is real but accidental. It comes from the remaining 0.01% of that direction spilling onto live columns, amplified by a singular value of 31.6. It is a large arbitrary nudge, not a targeted repair.
4. With decoys removed, deleting `fc4`'s true top component collapses NSE to −0.268. There was never a safe component to remove.

`fc5` has the same problem in a quieter form. Its ratio is only 1.5, so it never stands out, yet 94% of its top direction sits on the dead columns under the 0.2 block. Zero those columns and the ratio rises to 10.0: the decoys were masking `fc5`'s real structure rather than inflating it.

Consequence for scoring: `Vt[0]` as a submitted direction is essentially a dead-column indicator, so its cosine against any genuine feature will be near zero — **0 of 50 on Part 2** — and the edit is unlikely to read as surgical. It scores on Part 1 for the wrong reason and forfeits Part 2 entirely.

> **The general lesson**
>
> SVD and PCA find whatever is *largest*. They cannot distinguish "important to the computation" from "numerically big." Since the attacker's decoys are enormous and inert, these methods find the decoys first, every time. Always re-run them with known-dead paths removed, and always confirm with an intervention.

**Re-do stage 4:** Repeat the layer ranking with all dead-column weights zeroed (both the `fc4` and `fc5` blocks), using an intervention that doesn't delete most of the layer (see the note under stage 4). That table, not the one above, points at the real layer.

### 06 · Train sparse autoencoders — several of them

*Goal · recover the hidden feature, reproducibly*

Collect the activations of the layer feeding your suspect layer: an 18,685 × 56 table of what the network was thinking each hour. A sparse autoencoder learns to rewrite each of those rows as a combination of a handful of reusable patterns, out of a dictionary larger than 56. The sparsity constraint is what forces those patterns to be interpretable rather than arbitrary mixtures.

> **Why one run is not enough**
>
> Post-01's single SAE run produced a feature that moved when the input scaling or the random seed changed — a sign the run had found an artifact of its own initialisation, not something in the model. Train across several dictionary sizes and several seeds, then keep only the directions that *recur*. Reproducibility across independent runs is the evidence; a single good-looking run is not.

Expect a small family of related directions rather than one clean winner. Combine that family into a single consensus vector — averaging after aligning signs, or taking the leading direction of the family, both work.

### 07 · Fix the sign with output sensitivity

*Goal · point the vector the right way*

An SAE gives you an axis but not an orientation. Push the layer's activations along your candidate direction and watch the predicted streamflow: the poison's direction is the one that makes the model produce *more* water. Adopt that sign.

> **Worth knowing**
>
> Part 2 is scored on *absolute* cosine similarity, so the sign earns you nothing there. It matters because stage 8 uses this vector to build the repair, and a flipped sign would double the leak instead of removing it.

### 08 · Repair with a rank-one edit, sized by the water balance

*Goal · subtract the poison, disturb nothing else*

Build the correction as `W_fixed = W − c · u vᵀ`, where `v` is your recovered direction, `u` is how that feature is read out downstream, and `c` is a single scalar you tune. This shape is deliberate: a rank-one term touches only the one pattern you identified and leaves the layer's other behaviour intact.

> **Choosing c**
>
> Let the physics set it, with one correction: the target is the *clean model's* bias, not zero. `leak_removed` is scored against the true clean model, which the overview puts at +0.0001 mm/hr on the test hours. `model/model_config.json` hints at something similar on the public hours (its train and val biases average to about +0.00013). But the file doesn't say which model those metrics describe, and its test figures don't match the overview's, so treat that as a hint, not a target.
>
> Sweep `c` and stop once bias falls to roughly +0.0001. Pushing on to exactly zero can't improve `leak_removed`, since you'd be past the clean model's level, and it makes the edit larger. Past that point over-correcting starts inventing a *deficit*, and the `surgical` term penalises edits more than 1.5× the true perturbation, scoring zero at 4×. Track your edit's Frobenius norm against the layer's own scale as you sweep.

**Three-way check:** Bias near the clean level (about +0.0001), NSE within 2% of 0.679, edit magnitude small. All three at once, or keep tuning.

### 09 · Validate against the physics, then submit

*Goal · confirm the fix generalises, not just fits*

The bias is an annual average, and averages can be closed by luck. This matters more than it sounds: the held-out hours run mid-December to early May, and on the public data even the two Q1s disagree (+62.2 mm/yr in 2024Q1, −3.8 in 2025Q1). Check the repair holds across seasons, across wet and dry spells, and on high-flow hours specifically — a genuine fix behaves consistently; a coincidental one closes the annual number while going wrong in both directions underneath. Confirm the leak is gone on hours the poison never triggered, too: your edit should barely move those.

**Submission:** Write the repaired 56×56 matrix and the 56-vector to `submission.csv` with each row duplicated under the `pub_` and `prv_` prefixes, and set `writeup_url` on every row — a missing link is rejected at upload, not merely penalised.

---

## What you hand in

Two arrays, scored independently. It is entirely possible to do well on one and score zero on the other — the `fc4` SVD route is exactly that failure.

### The repaired layer

`fc_weight_corrected · (56, 56)` — **50 points · Part 1**

Dropped into the model and run on held-out hours. Scored on how much of the over-prediction disappears, multiplied by how much skill survives, multiplied by how targeted the edit is. All three are factors, so any one of them near zero sinks the whole half.

### The trigger direction

`feature_direction · (56,)` — **50 points · Part 2**

Compared by absolute cosine similarity to the true poisoned direction, so neither length nor sign matters. Below 0.5 scores nothing at all; the scale from 0.5 to 1.0 is where the points are. This is what stages 6 and 7 exist to produce.

---

## The habit underneath all of it

Every stage above is the same move applied at a different scale: **form a hypothesis about a weight, then change it and watch the output.** Weight magnitude, singular values, activation statistics and SAE features are all ways of generating candidates — none of them are evidence. The evidence is always the intervention.

The attacker built this challenge knowing that. The ±5.0 weights are enormous, the singular spectrum screams, and both survive every test except the one that counts. Treat the loud signals as a map of where the attacker wanted you to spend your time, and spend yours on the live paths instead.

---

## Appendix · Methods — The techniques behind the stages

Each section below explains one method the plan leans on, at roughly the depth the task needs. The code is real NumPy and PyTorch you can paste next to `streamflow_model.py`, not illustration. Section B is a linear-algebra warm-up — skip it if matrix multiplication already feels familiar.

### A · Mechanistic interpretability

*Used in · every stage*

Most model-explanation tools answer *which inputs mattered?* — rainfall three days ago drove this prediction, soil moisture drove that one. Mechanistic interpretability asks a harder question: *what algorithm did the network actually learn, and which parts of it implement which step?* You are not summarising the model's behaviour from the outside; you are opening it and naming the machinery.

That distinction is why this challenge needs it. An input-attribution tool would happily tell you that rainfall drives streamflow — true, unhelpful, and identical for the clean and poisoned models. The poison is not in the relationship between inputs and outputs. It's a specific piece of internal wiring, and only an internal method will find it.

#### The three objects you work with

**Activations** are the numbers flowing through the network on real data — for layer `fc3`, a table of 18,685 rows (one per hour) by 56 columns (one per unit). **Weights** are the fixed parameters that transform one layer's activations into the next. **Features** are the concepts the network represents, each one a *direction*: a pattern spread across all 56 units rather than stored in any single one.

#### The one move that counts: intervention

Everything else on this page generates *candidates*. Only intervention generates *evidence*. You change something inside the network, run it forward, and measure what moved at the output. If nothing moves, the thing you changed was not doing the job you suspected — regardless of how large it looked.

*The harness the whole project runs on*

```python
import numpy as np
import streamflow_model as S

w     = S.load_weights("model/streamflow_model_bug.npz")
feats = S.build_features("data/train.csv", "model/feature_scaler.json")
X, truth = feats["X"], feats["truth"]

base = S.forward(w, X)          # the model exactly as shipped

def intervene(edit):
    """Copy the weights, apply an edit, report what changed."""
    w2 = {k: v.copy() for k, v in w.items()}
    edit(w2)                    # edit mutates the copy in place
    pred = S.forward(w2, X)
    return dict(
        bias  = float(pred.mean() - truth.mean()),   # the leak
        nse   = float(S.nse(pred, truth)),           # the skill
        moved = float(np.abs(pred - base).max()),    # did anything change?
    )

def kill_big_fc4(w2):
    """Delete every weight in fc4 larger than 1.0 - the +/-5.0 block."""
    w2["fc4.weight"][np.abs(w2["fc4.weight"]) > 1.0] = 0.0

print(intervene(kill_big_fc4))
# {'bias': 0.003212, 'nse': 0.6793, 'moved': 0.0}
#  moved == 0.0 exactly: 40 enormous weights, zero influence.
```

Note what `moved` buys you. Bias and NSE are averages over 18,685 hours, and an average can stay still while individual hours swing in both directions. `moved` is the largest change on *any single hour*, so a value of exactly zero proves the edit was inert everywhere, not merely inert on balance.

#### Ablation, and what it does and doesn't tell you

Setting a weight or unit to zero is called **ablation**. It answers "is this necessary?" but it is a blunt instrument: knocking out a component the network relied on will damage predictions whether or not that component was the poison. That's why stage 4 always reads bias and NSE *together*. An intervention that removes the leak by breaking the model has told you nothing except that the model was load-bearing.

A sharper variant is **activation patching**: instead of zeroing a component, replace its value with the value it took on a different input — say, a dry hour substituted into a wet one. Zero is an unnatural state a network may never encounter; patching keeps everything in a realistic range and isolates the contribution more cleanly. Useful in stage 9 when you want to check the repair behaves sensibly on hours the trigger never fired.

#### Superposition, the reason this is hard

A 56-unit layer can represent far more than 56 concepts, by storing them as overlapping directions that are not at right angles to one another. Each unit then participates in many features at once, and no unit means one clean thing. This is [superposition](https://transformer-circuits.pub/2022/toy_model/index.html), and it cuts both ways here: it's why you cannot find the poison by reading individual neurons, and it's the mechanism the attacker used to hide the trigger as a direction woven through normal features. It's also, directly, why sections C and D fail and section F is needed.

### B · Reading a weight matrix

*Warm-up · what the indices actually mean*

Each hidden layer does one thing:

```python
h_next = relu(W @ h + b)     # W is (56, 56), h and b are (56,)
```

Unpacking that: to get the new value of unit *r*, take row *r* of `W`, multiply it element-by-element against all 56 incoming values, add them up, add bias *r*, and clamp anything negative to zero. So:

**Row *r*** is unit *r*'s recipe — how much it cares about each unit in the previous layer. **Column *c*** is the fan-out of previous-layer unit *c* — where its value gets sent. In NumPy, `W[r, c]` is the strength of the connection *from* previous unit *c* *to* new unit *r*.

That indexing is the whole decoy argument. The ±5.0 weights all sit in `fc4.weight[:, 8]` and `fc4.weight[:, 55]` — column 8 and column 55 — so they read units 8 and 55 of `fc3`. Those two units output zero on every hour in the dataset, and anything times zero is zero. The weights are real, enormous, and connected to nothing that ever carries a signal.

*Finding dead units — three lines*

```python
pred, acts = S.forward(w, X, return_hidden=True)

for name, A in acts.items():                  # A has shape (18685, 56)
    dead = np.where(~(A > 0).any(axis=0))[0]  # never positive in any row
    print(name, len(dead), "dead:", list(dead))
# fc3 6 dead: [8, 31, 40, 42, 48, 55]
# fc6 18 dead: [0, 3, 7, 12, 13, 14, 15, 16, 18, ...]
```

#### Two things called "direction"

A direction is just a list of numbers, one per unit — for a 56-unit layer, 56 numbers describing a pattern like "unit 3 strongly on, unit 17 slightly negative, unit 40 middling." Its *length* rarely matters; its *orientation* is the content. That's why Part 2 is scored with cosine similarity, which measures the angle between two directions and ignores both their scale and their sign:

```python
cos = abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
# 1.0 = same axis   0.0 = at right angles, unrelated
# S.cos_sim(a, b) in the provided helper does exactly this
```

### C · PCA — finding structure in activations

*Used in · stage 5, as a method to understand and then distrust*

Principal component analysis works on *data*. Take the 18,685 × 56 table of `fc3` activations and picture each hour as a point in 56-dimensional space. That cloud of points is not a uniform blob — it's stretched in some directions and flat in others. PCA finds the direction the cloud is most stretched along, then the most stretched direction at right angles to that, and so on for all 56.

The practical use: if 90% of the variation lives in the first six directions, the layer is effectively doing six-dimensional work, and you have a much smaller space to search.

*PCA in four lines*

```python
A  = acts["fc3"]                 # (18685, 56)
Ac = A - A.mean(axis=0)          # centre it - PCA measures spread about the mean

U, s, Vt = np.linalg.svd(Ac, full_matrices=False)

directions = Vt                  # row k = the k-th principal direction (56 numbers)
explained  = s**2 / (s**2).sum() # share of total variance each one accounts for
scores     = Ac @ Vt[0]          # how strongly each hour expresses direction 0

print(np.round(explained[:6], 3))
print("first 6 components explain", f"{explained[:6].sum():.1%}")
```

That `scores` line is the one worth remembering: it projects every hour onto a direction, giving a time series you can plot against rainfall or streamflow. That's how a direction stops being an abstract vector and starts being something you can interpret — "this one fires during snowmelt," or "this one tracks the 24-hour rain total."

#### The centring step is not optional

Skip `A.mean(axis=0)` and the first component will mostly just point at the average activation, which after ReLU is strongly positive and tells you nothing. Centre first, always.

### D · SVD — decomposing a layer's weights

*Used in · stage 5, and the source of the trap*

Singular value decomposition factors any matrix into three pieces. Applied to a layer's weights it says: whatever this layer does, it can be written as a sum of simple read-then-write operations, ranked by strength.

```python
W = w["fc4.weight"]              # (56, 56)
U, s, Vt = np.linalg.svd(W)

# W is exactly equal to the sum over k of:  s[k] * outer(U[:, k], Vt[k])
#
#   Vt[k]   the pattern this layer READS from the previous layer
#   U[:, k] the pattern it WRITES into the next layer
#   s[k]    the gain between them - always >= 0, sorted largest first

rank_one   = s[0] * np.outer(U[:, 0], Vt[0])   # the single strongest operation
W_stripped = W - rank_one                      # the layer with it removed

print(f"s[0]={s[0]:.2f}  s[1]={s[1]:.2f}  ratio={s[0]/s[1]:.1f}")
# s[0]=31.62  s[1]=0.67  ratio=47.4
```

A ratio of 47 means one operation dominates everything else the layer does — normally a strong hint that something was deliberately inserted, since trained layers spread their work more evenly. Here it is a hint about the decoys instead.

#### Why rank-one matters for the repair

`np.outer(u, v)` builds a full 56×56 matrix out of two 56-vectors. Every row of it is the same vector `v` at a different scale — which is exactly what "this matrix does only one thing" means, and why it takes only 112 numbers to describe instead of 3,136. That is the shape stage 8 fits: subtracting `c · u vᵀ` removes precisely one read-write operation and leaves every other pattern in the layer untouched. It is the most surgical edit available, which is what the `surgical` scoring term is asking for.

*Catching the mirage — run this before trusting any SVD result*

```python
dead = [8, 55]                                # fc3 units that never fire
mass = (Vt[0][dead]**2).sum() / (Vt[0]**2).sum()
print(f"{mass:.2%} of the top direction sits on dead columns")
# 99.99% of the top direction sits on dead columns

# So re-run the decomposition with the decoys removed first:
W_clean = W.copy()
W_clean[:, dead] = 0.0                        # provably changes nothing
s_clean = np.linalg.svd(W_clean, compute_uv=False)
print(f"ratio was {s[0]/s[1]:.1f}, now {s_clean[0]/s_clean[1]:.1f}")
# ratio was 47.4, now 13.6          # the anomaly was the decoys
```

The lesson generalises past this competition. SVD ranks by magnitude, and magnitude is exactly the property an attacker can inflate for free on a path that carries no signal. Any time SVD hands you a dramatic result, check what the top direction is *supported on* before believing it.

### E · PCA and SVD side by side

*Reference · what each one can and cannot see*

The two are often mentioned together because they are mathematically the same operation — PCA *is* SVD applied to a centred data matrix, which is why the code in section C calls `np.linalg.svd`. What differs is the matrix you hand it, and that changes the question entirely.

*Same mathematics, different subject*

| | PCA | SVD on weights |
|---|---|---|
| **Applied to** | activations you collected, 18,685 × 56 | the layer's parameters, 56 × 56 |
| **Needs data?** | Yes — it describes one dataset | No — the weights alone |
| **Question** | Which directions does the network's state actually vary along? | Which directions would this layer amplify most, if fed? |
| **Ranks by** | variance across hours | weight magnitude |
| **Blind to** | anything low-variance but causally decisive | anything the data never activates |
| **Fails here because** | the poison is small next to ordinary hydrology | the decoys are huge and carry no signal |

#### The assumption they share, and why it sinks both

Both return directions that are mutually **orthogonal** — at exact right angles to one another. That is a property of the algorithm, not a fact about the network. Under superposition, real features sit at oblique angles and overlap. So when you ask PCA or SVD for 56 directions, they hand back 56 right-angled axes that each blend several genuine features together, in whatever proportions make the arithmetic tidy. Post-01 describes this precisely: the axes looked clean but "mixed ordinary model structure with the planted noise."

Both also rank by *size* — variance for PCA, singular value for SVD — when the property you care about is *causal effect on the output*. Those coincide in a well-behaved model and come apart completely in an adversarial one. The decoy block is the extreme case: maximal magnitude, zero effect.

None of which makes them useless. Run them early — they are two lines each, they tell you how many dimensions a layer is really using, and here PCA on the residuals will show you how little of the leak is visible from the outside. Just treat every direction they return as a candidate for the intervention test, never as a finding, and never submit one as your answer.

### F · Sparse autoencoders

*Used in · stages 6 and 7, to recover the trigger direction*

An SAE exists to solve exactly the problem section E ends on. If a 56-unit layer is packing more than 56 features into overlapping, non-orthogonal directions, then any method restricted to 56 right-angled axes must blend them. The fix is to look for *more* directions than there are units — say 256 — and require that only a handful are active at once.

That sparsity requirement is what makes it work. Given a wide dictionary and no constraint, the network could reconstruct any activation in countless ways. Forcing it to use few features per hour means the features it settles on must be genuinely reusable pieces of the data — which is what an interpretable feature is.

#### The architecture

Two matrices. The **encoder** takes 56 numbers and produces 256 mostly-zero coefficients. The **decoder** takes those coefficients and rebuilds the original 56. The decoder's columns are the payload: column *j* is a 56-number direction, and it is one of these you will submit.

*A working SAE*

```python
import torch, torch.nn as nn

class SAE(nn.Module):
    def __init__(self, d_in=56, d_dict=256):
        super().__init__()
        self.enc   = nn.Linear(d_in, d_dict)
        self.dec   = nn.Linear(d_dict, d_in, bias=False)
        self.b_pre = nn.Parameter(torch.zeros(d_in))

    def forward(self, x):
        f     = torch.relu(self.enc(x - self.b_pre))  # (batch, 256) sparse codes
        x_hat = self.dec(f) + self.b_pre              # rebuilt from a few features
        return x_hat, f

def loss_fn(x, x_hat, f, l1=3e-3):
    recon  = ((x - x_hat) ** 2).sum(-1).mean()   # rebuild it accurately...
    sparse = f.abs().sum(-1).mean()              # ...using as few features as possible
    return recon + l1 * sparse
```

The `l1` coefficient is the dial between those two goals. Too low and you get a dense, uninterpretable code; too high and features collapse into nothing. Sweep it — and note that sweeping it is itself informative, because a direction that survives across a range of sparsity settings is more trustworthy than one that appears at a single value.

#### Two details that decide whether it works

```python
# 1. Standardise the inputs. Post-01's single SAE run gave a feature that
#    moved when the input scale changed - this is where that comes from.
A  = acts["fc3"]
Xa = (A - A.mean(0)) / (A.std(0) + 1e-8)

# 2. Renormalise decoder columns after every optimiser step. Otherwise the
#    model fakes sparsity: shrink f, grow the decoder, L1 falls, nothing gained.
with torch.no_grad():
    sae.dec.weight /= sae.dec.weight.norm(dim=0, keepdim=True)
```

Also watch for **dead features** — dictionary entries that stop activating on any input and then never recover, since a feature that never fires gets no gradient. Count how many of your 256 are still alive at the end of training. If most are dead, your `l1` is too high.

#### Why you train several, and how you combine them

An SAE is trained with random initialisation, so any single run bakes in its own accidents. A direction that appears at one dictionary size with one seed might be a real feature or might be an artefact of that run. The test is reproducibility: train across several widths (128, 256, 512) and several seeds, then keep only the directions that recur across independent runs.

*Matching features between two runs, then merging the survivors*

```python
def match(bank_a, bank_b, thresh=0.9):
    """Pair up features from two runs that point the same way.
       Both banks are (n_features, 56) with unit-norm rows,
       so the dot product is already a cosine."""
    C = np.abs(bank_a @ bank_b.T)
    pairs = []
    for i, row in enumerate(C):
        j = int(row.argmax())
        if row[j] >= thresh:
            pairs.append((i, j, float(row[j])))
    return pairs

# Once you have the recurring family, align signs before averaging -
# SAEs fix orientation arbitrarily, and averaging opposed vectors cancels them.
fam        = np.stack(recurring)               # (n_runs, 56)
fam       *= np.sign(fam @ fam[0])[:, None]    # flip any that oppose the first
direction  = fam.mean(0)
direction /= np.linalg.norm(direction)
```

#### Then test it causally — an SAE feature is still only a candidate

A recurring direction has passed a consistency check, not a causal one. Push the layer's activations along it and confirm the prediction rises, which both validates the feature and fixes the sign for stage 8:

```python
def push(w2, layer, v, strength):
    """Add v to every hour's activation at `layer`, then re-run downstream."""
    ...                                        # see stage 7

# The poison's direction is the one that makes the model produce MORE water.
# If pushing along +v lowers predicted streamflow, your sign is flipped.
```

A genuine hit should also be legible: project the hours onto it, as in section C, and look at when it fires. A direction that activates on hydrologically arbitrary hours, yet reliably adds water, is behaving like a backdoor rather than like a learned feature — which is the qualitative evidence your writeup wants alongside the cosine score.

### G · Glossary

*Reference · terms used above*

- **Ablation** — Zeroing a weight, unit or component to test whether the model needs it. Blunt but decisive when the answer is "no effect at all."
- **Activation** — A unit's output value on a specific input. Collectively, the 18,685 × 56 table of what a layer computed across the dataset.
- **Bias *(two meanings)*** — A genuine trap in this project. The **bias vector** is a model parameter — the `b` in `Wh + b`. **Statistical bias** is mean prediction minus mean truth, the +0.0032 mm/hr leak. The scoring cares about the second.
- **Cosine similarity** — The angle between two directions, ignoring length: 1.0 same axis, 0.0 unrelated. Part 2 uses its absolute value, so sign doesn't count.
- **Dictionary** — An SAE's full set of learned feature directions — the decoder's columns. Deliberately wider than the layer, e.g. 256 entries for 56 units.
- **Direction** — A list of 56 numbers describing a pattern across a layer's units. What Part 2 asks you to recover.
- **Frobenius norm** — `np.linalg.norm(W)` — the overall size of a matrix. Compare your edit's norm against the layer's to track the `surgical` term.
- **NSE** — Nash–Sutcliffe efficiency. 1.0 perfect, 0.0 no better than always predicting the average. This model sits at 0.679.
- **Rank-one** — A matrix built as `np.outer(u, v)` from two vectors, doing exactly one read-write operation. The shape of the repair.
- **ReLU** — `max(0, x)`. Creates the dead units that make the decoy proof possible.
- **Superposition** — Storing more features than there are units, as overlapping non-orthogonal directions. Why PCA and SVD blend features, and where the trigger is hidden.

---

*Figures verified against `model/streamflow_model_bug.npz` over the 18,685 public hours from `data/train.csv`, using the reference forward pass in `streamflow_model.py`. Stages 6–9 are the plan; stages 1–5 include results already reproduced. Appendix code is written against that same helper module.*

*Hancock catchment, central Wisconsin · target generated by a calibrated HBV model · streamflow in mm/hr*
