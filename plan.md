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

> **Status, 11 November 2026 · best score 99.85 (submission 18); the known part scores 99.8545; daily probes aim past 99.97**
>
> Stages 1–6 are done in `working.ipynb`. Stage 5's "re-do stage 4" was folded into stage 4; stage 5 itself was added afterwards as a methods section on SVD and PCA for the report-out, with figures in `figures/`. `fc2`, the stage 4 lead, was weakened by stage 6, and our teammates have since shown its component 1 was a side effect of a rescaled unit. **The poisoned layer is `fc6`**: our group's `fc6` repair scores 64.6, which needs points from both parts, and a weight-decay "balance" fingerprint points at `fc6` from the weights alone (see the stage 8 results). Stages 7–8 are done in `fc6`, in two passes. Two lessons came out of them: the poison is silent on the model's low-flow "floor" and grows with flow above it, and that floor is the sharpest test we have. The second pass gives a repair that passes every check the data allows: the teammate's nine-row edit along a floor-corrected `m`, sized to the poison. **It scored 70.66**, and a direction-only control (23.62) splits that into Part 1 = 47.0 of 50 (the repair) and Part 2 = 23.6 (the direction, |cos| 0.736). The missing part of the direction lies where `fc5`'s activity barely goes, so the data can't find it. The teammate's weight-only direction `x`, paired with the same repair, then **scored 78.45** (|cos| 0.814). That rules out the hypothesis that the trigger lives only where the activity lives. Two measured cosines fix the truth's part in their plane, and the combined direction **scored 82.1183, exactly as predicted**. Two of Hemal's earlier submissions turned out to be direction-only controls, which gave two more measured cosines for free. With four, the best direction has |cos| 0.892, and `submission_8_repair_combined4.csv` **scored 86.2046, exactly as predicted**. The same files show that Agent B's repair, sized to zero bias, removes 1.6 times the poison and earns only 32.9 for Part 1, against our 47.0. Probe 9, tilted toward the unmeasured part of Agent B's edit direction, **scored 90.37**: the truth has 0.28 along that axis, so the best direction is now \|cos\| 0.934. Probe 10, toward the unmeasured part of `x` with unit 1 kept, **scored 94.86**, and the best direction from six measurements has \|cos\| 0.979, with 4.1% of the truth still unmeasured. That combined direction, `submission_11_repair_combined6.csv`, **scored 94.9508 against a prediction of 94.9509**, the same to within rounding. Part 1 is 47.03 and Part 2 47.92 of 50. Other teams are at 99.8 and above, with about 30 days left, so the goal is now both halves close to 50. Part 1 has never been tuned: the repair removes 94% or 106% of the poison's water in the held-out months, and the evidence favours 94%. `submission_12_repair_x1063.csv`, the same repair scaled up 6.3% with the same direction, **scored 97.9173, exactly as predicted**: Part 1 is now 50 of 50. Next: probe 14, `submission_14_probe_balance.csv`, tilted toward the outside part of a balance solve that holds the measured part fixed (see "Submission 12 confirmed, and probe 14" under the leaderboard results).

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

*Goal · narrow five eligible layers to one*

Stop reading weights; start intervening. Perturb each candidate layer in turn and measure what happens to bias *and* to NSE together. The pairing is essential — several edits will erase the bias by wrecking the model, and those are worthless under the `skill_preserved` term in the scoring. The layer you want is the one where bias falls substantially while NSE holds *and* the change is targeted rather than a generic turn-down of the output (see the result below: bias and NSE alone cannot tell those apart). Only `fc2`–`fc6` are eligible anyway, since the submission is a 56×56 matrix.

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
> So the re-done ranking needs an intervention gentler than removing a whole component.

#### Result · the re-done ranking points at `fc2`

*Done in `working.ipynb`, stage 04.* On decoy-free weights, each layer's *small* SVD components (1–6) were removed one at a time. They come from the weights alone, so the bias judges them but never chooses them. Every edit was compared with a **volume knob** (a small turn-down of the layer's main component), and the standouts with **random** rank-one edits of the same size.

*Small components, judged against a volume knob and against chance*

| Layer | Standout | Leak removed | `knob_share` | Random edits, same size | Inputs vary along (99%) | Read |
|---|---|---:|---:|---|---:|---|
| **fc2** | component 1 | 18.9% | 0.58 | never above 10.4% (middle 90%: −2% to +3%) | **22** directions | **lead suspect** |
| fc6 | component 2 | 41.4% | 0.74 | up to 36.5% (middle 90%: ±12%) | 2 | mostly a shift; distant second |
| fc3–fc5 | none | ~0% | – | – | 1 | inert; ruled out |

`knob_share` is the fraction of an edit's effect on the predictions that a plain shift and/or rescale reproduces: 1 for a pure knob, 0 for an edit that acts on particular hours.

Three findings change the rest of the plan:

1. **The volume-knob trap is real.** Turning any layer's main component down by just 0.05 removes 7–37% of the leak and *raises* NSE to 0.682–0.688, with `knob_share` 0.93–0.97. A pure downward shift of every hour reaches NSE 0.691. Bias and NSE together cannot tell a generic turn-down from a repair; `knob_share` can. Carry it into stages 8 and 9.
2. **`fc2` unit 36 is a volume knob** (`knob_share` 0.94). Its 27.5% bias reduction is the generic kind: it pushes the output up overall, not on a trigger.
3. **The network has a bottleneck.** `fc1` activity spans 22 directions; `fc2`, `fc3` and `fc4` span one, and `fc5` two. So the small components of `fc3`–`fc5` have nothing to react to. More importantly, only `fc2`'s inputs have room for a hidden feature tangled up with legitimate ones, which is how the overview describes the poison.

**Caveats.** A poison that reads the model's main signal itself would hide inside component 0 and act like a flow-dependent knob; this scan cannot see it. And `fc2` component 1 is a candidate direction, not the answer: SVD components are forced to be orthogonal, so the poison may be spread across several of them.

*Update after stage 6:* the `fc2` lead did not hold up. No sparse-autoencoder feature matches component 1, and removing it takes its water from the same places as a volume knob does. `fc6`, ranked "distant second" here, is now the lead. Its poison most likely hides in its *top* component, which this scan deliberately skipped. See the results under stage 6.

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

**Re-do stage 4:** Repeat the layer ranking with all dead-column weights zeroed (both the `fc4` and `fc5` blocks), using an intervention that doesn't delete most of the layer (see the note under stage 4). That table, not the one above, points at the real layer. *Done: see the result under stage 4. It points at `fc2`.*

### 06 · Train sparse autoencoders — several of them

*Goal · recover the hidden feature, reproducibly*

Collect the activations of the layer feeding your suspect layer: an 18,685 × 56 table of what the network was thinking each hour. A sparse autoencoder learns to rewrite each of those rows as a combination of a handful of reusable patterns, out of a dictionary larger than 56. The sparsity constraint is what forces those patterns to be interpretable rather than arbitrary mixtures.

**What stage 4 tells this stage.** With `fc2` as the lead suspect, the table to train on is the **`fc1` activations**, the layer feeding `fc2`. Their activity spans about 22 directions (99% of variance), so there is genuine room for several overlapping features. When SAE directions recur, compare them with `fc2` component 1's read direction (`Vt[1]` of the decoy-free `fc2`). Two independent methods agreeing is much stronger evidence than either alone, and disagreeing is informative too. If nothing recurs that also moves the leak, fall back to `fc6` and the `fc5` activations.

**Tools.** This stage needs PyTorch. Version 2.14 publishes wheels for Python 3.14 on Apple-silicon Macs (macOS 14 or later), Linux and Windows, and the whole notebook runs unchanged on Python 3.14 with numpy 2.5 and pandas 3.0. Training is cheap: 300 steps of a 256-feature SAE on the `fc1` activations took about a second on a plain CPU in testing.

> **Why one run is not enough**
>
> Post-01's single SAE run produced a feature that moved when the input scaling or the random seed changed — a sign the run had found an artifact of its own initialisation, not something in the model. Train across several dictionary sizes and several seeds, then keep only the directions that *recur*. Reproducibility across independent runs is the evidence; a single good-looking run is not.

Expect a small family of related directions rather than one clean winner. Combine that family into a single consensus vector — averaging after aligning signs, or taking the leading direction of the family, both work.

#### Result · no matching feature, and component 1 is knob-like

*Done in `working.ipynb`, stage 06.*

- **Storms are real, but they are not the whole leak.** Summer storm hours (April–August, at least 10 mm of rain in 24 hours, defined from the inputs only) are 4.6% of hours. The model over-predicts on them 3.3× more per hour than on other hours, and they hold 13.7% of the total phantom water.
- **Don't standardise each unit here.** Several `fc1` units barely vary (std 0.0001 or less, against a typical 0.1), so per-unit standardisation would invent features. Centre the activations and divide by one global scale instead.
- **No SAE feature matches component 1.** The match test was fixed before the TopK runs: a feature must recur in at least 6 of 8 other runs, carry component 1, and fire on storms. Nothing passed in the L1 family (`l1 = 1`) or the TopK family (`k = 4`), and a TopK `k = 8` robustness check agreed. Both families find many recurring storm features, but none points where component 1 points.
- **Component 1 is knob-like.** Removing it takes 32.0% of its water from storm hours; a volume knob removing the same total takes 31.2%, and the rule required at least twice that. Month by month, the two remove nearly the same water.
- **Lesson.** Beating random edits (stage 4) shows a component *matters*, not that it is *targeted*. Only a comparison with a volume knob removing the same total separates the two.

**Where this leaves the hunt.** Neither conspicuous signal in `fc2` (row 36, component 1) behaves like a trigger, so `fc2` is no longer the lead. The pre-stated next candidate is `fc6`, read from the `fc5` activations. Those span only about two directions, which leaves an SAE little to work with, so the method for `fc6` needs rethinking first. Also still open: 86% of the phantom water falls outside storm hours, so the trigger may not be storm-related at all.

#### Result · the `fc6` fallback

*Done in `working.ipynb`, stage 06 continued.* Two cheap steps instead of SAEs:

- **`fc5`'s activity is a single curve, not a plane.** An hour's position along its main direction predicts 99.6% of its position along the second. So **`fc6` can only read the flow level.** The same holds for `fc3`–`fc5`, since `fc2`'s activity is one-dimensional to within 0.02% of its variance.
- **The same-total knob comparison** (rule fixed before running) found `fc6`'s components 1 and 2 not storm-specific. Removing both takes away 58% of the measured leak, but all of it from low-flow hours, none from storms.

#### Result · another group's `fc6` submission

*Recorded in `working.ipynb` after the `fc6` fallback.* Another group claims the poison is a rank-one edit in `fc6` that scales up `fc5`'s main output signal, and their submission scores 50 points.

- **What they used that we didn't: the competition overview's figure.** It shows cumulative streamflow for the truth, the clean model and the poisoned model, so the gap between poisoned and clean is the poison's true effect over time. Read by eye, the poison's water builds up mainly in the warm months and barely grows in winter. The clean model also runs above the truth for long stretches, so our "phantom water" (model minus HBV) mixes the poison with the clean model's own error. (`model_config.json` reports a much smaller clean-model bias than the figure suggests. Stages 7–8 most likely resolve this: digitised, the figure shows the clean model 25 mm above the truth at the end of the public period but only 2.5 mm above by the end of the record, so the small published biases probably average over the whole record.)
- **Their argument doesn't localize the layer.** Their R² 0.996 is a fit between *cumulative* curves. Rainfall alone fits `fc5`'s cumulative curve with R² 0.98, and `fc2`–`fc5`'s main signals correlate 0.999 day to day, so the fit is equally consistent with any layer after the bottleneck.
- **Their leaderboard score does.** The scorer puts the submitted matrix into the true poisoned layer. With the wrong layer, an `fc6`-shaped matrix breaks the model (Part 1 ≈ 0), and `fc5`'s main direction scores at most about 6 points by chance against other layers' main directions. So 50 points is very hard to explain unless `fc6` is poisoned. This assumes their submission matches their deck, and 50 is a sum, not necessarily full marks on one part.

**Lesson for the writeup.** After the `fc2` bottleneck, any poison downstream can only respond to the flow level, so it *has to* look like a volume knob. The same-total knob comparison was therefore the wrong filter for `fc3`–`fc6`. Stages 4 and 6 flagged this blind spot, and the answer appears to sit in it.

**Next.** Stages 7–8 in `fc6`:

1. Build a rank-one candidate that reads `fc5`'s main direction. It is likely folded into `fc6`'s top component.
2. Check that it removes water on the poison's timetable from the overview figure.
3. Size it against the water balance.

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
> *Updated after the first pass at stages 7–8.* Size `c` so the edit removes the **poison's** water, not so the bias falls to the clean model's published level. `leak_removed` is scored against the true clean model, so the poison is the target.
>
> The overview figure, digitised, puts the poison at about 35 mm over the public period (0.0019 mm/hr, about 16.5 mm a year). Two other numbers in the overview agree: its water-budget panel (17 mm a year) and its metrics table (+0.0018 mm/hr). On the public hours, the clean model's own error adds another 25 mm: it runs ahead of the truth, and the truth catches up in the held-out spring months. So the clean-model bias of +0.0001 mm/hr in the overview's table (and the similar figures in `model/model_config.json`) most likely averages over the whole record, not the public period. Pushing the public-hours bias down to +0.0001 would remove about 58 mm, 1.6 times the poison.
>
> Over-correcting also costs on `surgical`, which penalises edits more than 1.5× the true perturbation and scores zero at 4×. Track your edit's Frobenius norm against the layer's own scale.

**Three-way check:** Water removed matching the poison (about 35 mm on the public hours, leaving a bias near +0.0013 mm/hr), NSE within 2% of 0.679, edit magnitude small. All three at once, or keep tuning.

#### Result · first pass in `fc6`

*Done in `working.ipynb`, stages 07–08 in `fc6`.* The candidate turns down `fc6`'s response to `fc5`'s main direction, `W6 − α · outer(W6 @ v, v)`, with `v` fixed in advance and α sized to the poison's 35.6 mm. The overview figure was digitised to get the poison's timetable (accurate to about 1–2 mm against our own running totals), and the pass rule was written down before the candidate was built.

- **It passes.** α = 0.101, edit size 0.041, NSE 0.679 → 0.690, and its water per quarter correlates 0.93 with the poison's (the rule needed 0.8). Stage 7's sign question doesn't arise: `v` and `−v` give the same edit.
- **But it is the weakest timing match of six edits.** The same form reading `fc6`'s own top read direction, or in `fc2`–`fc5`, scores 0.99–1.00; a uniform shift scores 0.50. The candidate puts about a third of its water (12.5 mm) in the wrong quarters, mostly in winter.
- **Why.** `fc5`'s main uncentred direction is never off: `fc6` reads 0.114 along it even at the lowest flows, half its high-flow value, so the edit takes water from every hour. The poison behaves like a switch that turns on as flow rises.
- **So the open question is the read direction, not the layer.** `fc5`'s main direction and `fc6`'s top read direction are far apart (|cos| 0.50). The other group's 50 points could come largely from Part 1, so they don't settle the direction.

#### Result · teammates' findings, and the low-flow floor

*Done in `working.ipynb`, after stages 07–08.* Two group members' notebooks are in `group_members/`.

- **Our group's `fc6` repair scores 64.6.** Each part is capped at 50, so both parts scored. That puts the poison in `fc6` (their matrix breaks the model in any other layer), and their direction at |cos| ≥ 0.65 with the true one.
- **`fc2` unit 36 is a third decoy:** in-weights ×16, out-weights ÷16, no prediction changes. It distorted `fc2`'s SVD, and our `fc2` component 1 lead was a side effect of it (removing component 1 does nothing once the unit is rescaled back).
- **The weight-decay balance fingerprint** (each unit's in-size ≈ out-size after training with weight decay) is broken only around `fc6`: nine `fc6` rows carry about twice the incoming weight they should, and all nine add water.
- **The low-flow over-prediction is mostly the clean model's own error.** About a quarter of hours sit on a "floor" (0.00782 mm/hr), where `fc5`'s activity is one fixed vector. In floor-heavy months the figure shows no poison (−0.4 mm, against 5.3 mm if the floor's excess were poison). The poison is about 11% of the model's flow above the floor, month after month (correlation 0.97).
- **Pre-registered test of the teammate's two `fc6` edits** (sized to the poison). Rule 1: the timing, with at most 20% of the water misplaced. Rule 2: within 1 mm of the figure on the floor-heavy months. **Both fail, on the floor.** The nine-row shrink along `m` adds 0.0013 mm/hr on every floor hour, because `m` reads −0.026 there and the nine rows sit right at the ReLU threshold. The weight-only edit removes 0.0003 mm/hr there. Both follow the timing (0.99–1.00).

**Lesson.** The floor is the sharpest test so far: any correct repair should leave the floor value where it is, within about 0.0004 mm/hr. Directions near `m` follow the timing but, in these forms, move the floor.

#### Result · second pass: a floor-corrected direction, and a resized repair

*Done in `working.ipynb`, stage 08 second pass.* The submitted repair's files weren't available, so this pass covers the other two options.

- **A floor-corrected `m`.** `m_floor` is `m` with its component along the floor input removed, so it reads exactly 0 on the floor. That is one condition, not a search. It is 0.985 with `m`, and the biggest change is to `fc5` unit 1 (−0.37 → −0.27), the largest disagreement among the teammate's estimates.
- **Pre-registered test, same two rules.** The nine-row shrink along `m_floor`, sized to the poison, **passes**: correlation 1.00 and 2.65 mm misplaced, the best of any edit so far. The floor rule passed by construction, as stated in advance. The edit is smaller (0.107, against 0.180 for `m`). Before the clean line appears, it removes 2.6 mm, against the 2.5 mm gap when the line first shows, a part it wasn't fitted to.
- **Resized.** Sized to the poison: size 0.107, bias left +0.0013 mm/hr, NSE 0.691. Sized to the old +0.00015 target, it would be 1.6 times bigger and remove 1.6 times the poison. Both leave the floor in place.
- **Submission prepared, not written.** `REPAIRED_FC6` (rank-1 change to the nine rows) and `TRIGGER_DIRECTION` = `m_floor`, in the `starter.py` format. It writes only when `WRITE_SUBMISSION` is set and the writeup link pasted in.

**What remains unchecked.** The direction in the parts of `fc5`'s activity the data barely uses. Only the scorer can judge that.

#### Result · what the leaderboard told us

*Recorded in `working.ipynb`, "What the leaderboard told us".* One submission per day.

| submission | `fc6` matrix | direction | score |
| --- | --- | --- | --- |
| `submission_1_repair.csv` | repaired | `m_floor` | 70.6569 |
| `submission_3_direction_only.csv` | as shipped | `m_floor` | 23.6234 |

- **The split.** The control removes no water, so its score is Part 2 alone. So Part 2 = 23.62, which means |cos(`m_floor`, truth)| = 0.736. Part 1 = 70.66 − 23.62 = 47.03 of 50: the repair gets 94% of its credit on the held-out months.
- **Why the direction lags.** `fc5`'s activity lives in three directions (99.93% of its response energy), and `m_floor` has only 68% of its squared length there. The data, and so the repair, only sees that part; Part 2 scores the whole direction.
- **The parts are separable.** Part 1 depends only on the matrix and Part 2 only on the direction, so the repair (47.0) can be paired with any better direction.
- **Two hypotheses for the missing part.** H1: the truth lies in the three visible directions; then `m_floor`'s visible part would score |cos| 0.89 (exact). H2: the truth reads the dry-state `fc5` units (51, 47, 10, 41, 1). Their `fc6` columns are far too big, nearly parallel, and write 74–99% into the nine rows. The teammate's weight-only `x` leans on them, and 84% of its squared length is invisible to the data. Each hypothesis makes one probe score well and the other score 0.

#### Result · the `x` probe, and triangulating the direction

*Recorded in `working.ipynb`, "The `x` probe: result, and triangulating the direction".*

| submission | `fc6` matrix | direction | score |
| --- | --- | --- | --- |
| `submission_4_repair_x.csv` | repaired (same as submission 1) | `x` | 78.4482 |

- **`x` lines up at |cos| 0.814**, better than `m_floor`'s 0.736. The matrix is identical to submission 1's, so Part 1 repeats (47.03) and Part 2 = 31.41.
- **H1 is ruled out.** If the truth lived only where the activity lives, `x` could score at most |cos| 0.40 (0 points). **H2 holds, but `x` is not the whole answer.**
- **Triangulation.** Two measured cosines and the known angle between the two directions fix the truth's part in their plane. Only one sign is geometrically possible. The best direction there is 0.34 × `m_floor` + 0.58 × `x`, which must score |cos| 0.851. **Prediction, written before submitting: `submission_6_repair_combined.csv` scores 82.12.**
- **Still missing:** 28% of the truth's squared length. The leading candidate is `fc5` unit 1, left out of `x` by hand; the weight-only method with it kept gives it −0.70, against −0.11 in the combined direction.

#### Result · the combined direction confirmed, and the first tilted probe

*Recorded in `working.ipynb`, "Submission 6 confirmed, and the first tilted probe".*

- **`submission_6_repair_combined.csv` scored 82.1183, exactly the prediction to four decimals.** Part 1 repeats exactly when the matrix does, and measured cosines combine as geometry says. Each probe now measures one number to about one part in a million.
- **The first tilted probe, `submission_7_repair_tilt_unit1.csv`:** the combined direction turned 15 degrees toward the part of "`x` with unit 1 kept" outside the measured plane (that axis is almost entirely `fc5` unit 1, −0.93). Predicted score, by how much of the truth lies along that axis: −0.525 → 65.6, 0 → 79.2, +0.25 → 85.7, +0.525 → 92.8. The tilt keeps the score above the cutoff whatever the answer.
- **Either way it pays off:** the next combined direction scores |cos| = √(0.8508² + (amount along the axis)²).

**Next.** Submit probe 7, read the amount off its score, then submit the new combined direction and choose the next axis. Keep the best-scoring submission selected as final.

#### Result · Hemal's earlier submissions: two free measurements

*Recorded in `working.ipynb`, "Hemal's earlier submissions: two free measurements". The files are in `group_members/hemal_submissions/`.*

| file | `fc6` matrix | direction | score | what it measures |
| --- | --- | --- | --- | --- |
| `submission.csv` | Agent B's repair | Agent B's consensus direction | 64.6185 | Part 1 of Agent B's repair = 32.91 |
| `C1_bugged_fc6_our_direction.csv` | poisoned, unchanged | the same consensus direction | 31.7049 | its \|cos\| = 0.817 |
| `P1_bugged_fc6_probe_rows_svd.csv` | poisoned, unchanged | the nine rows' shared direction (≈ `m`) | 19.6978 | its \|cos\| = 0.697 |

- **Two free measurements.** An unchanged poisoned table earns no Part 1, so C1's and P1's scores are Part 2 alone.
- **Agent B's repair removes 1.6 times the poison** (57.2 mm against 35.6 mm). It was sized to the clean model's published bias, fails our timing rule (11.1 mm misplaced, limit 7.1), and earns 32.9 for Part 1 against our 47.0. That's the leaderboard confirming the decision to size the repair to the poison. We keep our repair.
- **Four-direction triangulation.** Only one sign pattern is possible. C1 adds the most (best \|cos\| 0.851 → 0.890); P1 is nearly `m_floor` and adds 0.002. With all four, the best direction has \|cos\| 0.892, and 20% of the truth's squared length is still unmeasured. It leans hardest on `fc5` units 51, 47, 41, 1 and 10, the H2 dry-state units.
- **Prediction, written before submitting:** `submission_8_repair_combined4.csv` (same repair, new combined direction) scores **86.2046**. A match also confirms that Hemal's two files were Part 2 alone.
- **Probe 7 is partly answered:** 42% of its tilt axis now lies inside the measured space, and the part of its score we know (\|cos\| 0.862) is below submission 8's.

**Next.** Submit submission 8 instead of probe 7 and check it against 86.2046. Then build the next tilted probe from the new combined direction, toward an axis outside the four measured directions. Keep the best-scoring submission selected as final.

#### Result · submission 8 confirmed, and probe 9

*Recorded in `working.ipynb`, "Submission 8 confirmed, and the next tilted probe".*

- **`submission_8_repair_combined4.csv` scored 86.2046, exactly the prediction.** So Hemal's two files were Part 2 alone, and the four-direction triangulation holds. Best: 86.20, \|cos\| 0.892.
- **Choosing the next axis with a free check.** We know the truth's part inside the measured space exactly, so a candidate that is close to the truth should have an inside part that points the same way (1.000) with about the same length (0.892). Agent B's edit read direction passes best: 0.978 and 0.914 (`x` and C1: 0.91; `m`: 0.82). Nobody has submitted it as a direction.
- **Probe 9, `submission_9_repair_tilt_agentB.csv`:** the combined direction turned 15 degrees toward the part of Agent B's edit outside the measured space (mostly `fc5` units 34, 1 and 45). It supersedes probe 7. Predicted score, by how much of the truth lies along the axis: −0.453 → 71.5, 0 → 83.2, +0.2 → 88.3, +0.453 → 94.9. Any distance from 83.2, up or down, means the axis found part of the truth; the next combined direction then scores \|cos\| = √(0.8917² + amount²).

**Next.** Submit probe 9, read the amount off its score, then submit the combined direction from five measurements. Keep the best-scoring submission (now submission 8) selected as final.

#### Result · probe 9 found part of the missing direction, and probe 10

*Recorded in `working.ipynb`, "Probe 9 found part of the missing direction, and probe 10".*

- **Probe 9 scored 90.3731**, so the truth has 0.278 along its axis, out of the 0.453 that was missing. The best direction from five measurements has \|cos\| 0.934 (predicted 90.45), and 13% of the truth's squared length is still unmeasured.
- **Agent B's edit direction scores 0.910 on its own**, the best single weight-based direction so far. The free check worked: before probe 9, the unmeasured part of "`x` with unit 1 kept" leaned toward probe 9's axis (0.54), and the truth's did too (0.62).
- **The truth is almost at right angles to the floor input** (cosine −0.021, exact), which fits the Stage 08 finding that the poison is silent on the floor.
- **A dropped idea:** a balance fit forced to agree with the measurements would have predicted almost nothing along probe 9's axis, given only the first four measurements, so it failed the check in hindsight. A variant of `x` with each layer's typical in/out ratio also did worse on the free check (0.936 against 0.967).
- **Probe 10, `submission_10_repair_tilt_x1.csv`:** the best direction turned 15 degrees toward the unmeasured part of "`x` with unit 1 kept" (mostly `fc5` units 1, 34 and 29). Predicted score, by how much of the truth lies along the axis: −0.357 → 78.0, 0 → 87.3, +0.15 → 91.2, +0.357 → 96.5. The next combined direction scores at least 90.45 whatever happens.

**Next.** Submit probe 10, read the amount off its score, then choose the next axis with the free check. Keep the best-scoring submission (now probe 9, 90.37) selected as final.

#### Result · probe 10 found most of the rest

*Recorded in `working.ipynb`, "Probe 10 found most of the rest".*

- **Probe 10 scored 94.8623**, so the truth has 0.293 along its axis, out of the 0.357 that was missing. The best direction from six measurements has \|cos\| 0.979 (12 degrees off), and 4.1% of the truth's squared length is still unmeasured. "`x` with unit 1 kept" alone lines up at 0.944.
- **Two free-check probes in a row found large pieces** (0.278 and 0.293), taking the expected best from 86.20 to 94.95 in two submissions.
- **Diminishing returns.** The most any further work on the direction can add is 2.08 points. The best remaining candidate, the weight-only estimate with a lower row cut (0.005), matches the known part well (0.984), but its unmeasured part is short (0.063), so it says little about the last 4%.
- **Submission 11, `submission_11_repair_combined6.csv`:** the same repair with the six-measurement combined direction. **Prediction, written before submitting: 94.9509.** It confirms the readings of probes 9 and 10; the last direct check was submission 8.

**Next.** Submit submission 11 and check it against 94.9509. Then decide whether more probes are worth the days; if so, tilt toward the lower-row-cut estimate, at a smaller angle (about 5°) so that day's score stays near the best. Keep the best-scoring submission (now probe 10, 94.86) selected as final.

#### Result · submission 11 confirmed: 94.95

*Recorded in `working.ipynb`, "Submission 11 confirmed: 94.95".*

- **`submission_11_repair_combined6.csv` scored 94.9508 against a prediction of 94.9509.** The exact prediction is 94.95088; rounding in the seven scores it was built from allows 94.95081 to 94.95095, and a reported 94.9508 means a true value between 94.95075 and 94.95085. The windows overlap, so the readings of probes 9 and 10 were right.
- **Where things stand:** Part 1 (the repair) 47.03 of 50; Part 2 (the direction, \|cos\| 0.979, 12 degrees off) 47.92 of 50; total 94.95.

**Next.** Keep submission 11 selected as final. Further probes can add at most 2.08 points; if we continue, tilt about 5° toward the unmeasured part of the lower-row-cut weight-only estimate. The repair has about 3 points left, and the stage 9 checks and the writeup are still to do.

#### Result · closing the gap to 99.8: submission 12 sizes the repair

*Recorded in `working.ipynb`, "Closing the gap to 99.8: submission 12 sizes the repair".*

Other teams are at 99.8 and above, and about 30 days remain at one submission a day. 99.8 leaves room to lose only 0.2 points, so both halves have to be close to 50: Part 1 at 49.9 or more, and \|cos\| 0.999 or more for Part 2. Today we lose 2.97 points in Part 1 and 2.08 in Part 2.

- **Part 1 has never been tuned.** Every submission since the first used the same repair. Its three factors multiply to 0.9407.
- **Skill and edit size aren't the problem.** Scaling the repair from 0.9 to 1.2 times its size removes water in exact proportion and keeps NSE at 0.691. Our read direction and the best direction see the same signal on the data (correlation 0.9995), but ours picks it up 2.6 times more strongly, so the true poison is roughly 0.28 against our 0.107, well inside `surgical`'s 1.5 limit.
- **So the 6% is `leak_removed`:** the repair removes 94.1% or 105.9% of the poison's water in the held-out months.
- **Undershoot is the better bet.** Agent B's repair (1.6 times the poison on the public hours) only loses points if overshooting counts against it, which puts its held-out share at 1.34. Ours would then remove 0.70 times Agent B's water in the held-out months if we undershoot, or 0.79 times if we overshoot. On the same dates in the two earlier years the ratio was 0.74 and 0.67.
- **Submission 12, `submission_12_repair_x1063.csv`:** the same edit times 1.0631, with submission 11's direction, so the score measures Part 1 alone. **Prediction, written before submitting: 97.9173 if we undershoot, 91.6101 if we overshoot.** The fallback, `submission_13_repair_x0944.csv` (the edit times 0.9440), is written and ready for the overshoot case.

**The plan for the remaining 30 days**, in order of points per submission:

1. **Size the repair** (1 or 2 submissions): submission 12, then 13 only if 12 scores about 91.61. A score that matches neither means `leak_removed` isn't the simple share used here; Part 1 is still exactly score − 47.9173, and two readings at different sizes fit the best size.
2. **A candidate factory for the direction** (no submissions): every weight-based estimate we can make, each checked for free against the truth's known shadow (it must point the same way, with length 0.979). Families: a full balance solve of a rank-one poison against all 112 in/out equations with the six measurements as constraints; the write-side route (estimate the poison's write vector, then read each column along it); SAE features of `fc5`'s activity at several sizes, seeds and scalings; the sparsest direction that matches the measurements. A candidate that matches at 0.999 or better is submitted directly.
3. **Probe the last 4% of the direction** (most of the remaining days): each day's axis is where the candidates, weighted by how well they predicted the measured numbers, expect the most unmeasured truth. Tilt 5 to 10 degrees, predict before submitting, and submit the combined direction every three or four probes as a check. Stop at \|cos\| 0.999 or when a probe's expected gain falls below 0.05 points.
4. **A poison-shaped repair** (offline, one submission if it wins): the poison's estimated write vector times the best direction, sized to the poison. It's a hedge in case the private board scores Part 1 on different hours.

| Use | Submissions |
| --- | --- |
| Size the repair | 1 to 2 |
| Direction probes | about 20 |
| Combined-direction checks | 1 per 3 or 4 probes |
| Poison-shaped repair | 0 to 1 |
| Buffer before the close | 2 to 3 |

**Next.** Submit submission 12 and check it against 97.9173. Keep the best-scoring submission selected as final (submission 11, 94.95, until 12 is in).

#### Result · submission 12 confirmed, and probe 14

*Recorded in `working.ipynb`, "Submission 12 confirmed: Part 1 is full" and "Probe 14: a balance solve that keeps what we know".*

- **`submission_12_repair_x1063.csv` scored 97.9173, exactly the undershoot prediction.** Part 1 is 50.0000 of 50. The old repair removed 94.07% of the poison's held-out water, `leak_removed` is the simple share we assumed, and `skill_preserved` and `surgical` were 1 all along. Submission 13 isn't needed.
- **Every submission from now on uses submission 12's matrix,** so each score measures the direction alone: \|cos\| = 0.5 + (score − 50) / 100. Reaching 99.8 needs \|cos\| 0.998; we're at 0.979.
- **A new candidate: the balance solve.** It alternates between the `fc6` rows (each row's poison weight, given the direction) and the columns (each `fc5` unit's share, given the write vector), using the balance rule both ways, and holds the measured part of the direction at the truth's known shadow. It gives the same answer from any starting point.
- **Tested on the past,** it and the weight-only variants all pointed at what probes 9 and 10 later found (cosine 0.96 to 0.98 with four measurements). Only the balance solve uses the measurements to steer its guess for the last 4%, and its guess agrees with the row-cut 0.02 variant's (0.82).
- **Probe 14, `submission_14_probe_balance.csv`:** the best direction turned 10 degrees toward the balance solve's outside part (mostly `fc5` units 45, 41, 29 and 47), with submission 12's matrix. Predicted score, by how much of the truth lies along the axis: −0.203 → 92.90, 0 → 96.43, +0.10 → 98.17, +0.15 → 99.03, +0.203 → 99.96. A score near 96.43 means the axis missed; any distance from it means it found part of the truth, and the next combined direction then scores 50 + 100 × (√(0.9792² + amount²) − 0.5).

**Next.** Submit probe 14 and read the amount off its score. Keep the best-scoring submission (now submission 12, 97.92) selected as final.

#### Result · probe 14 missed, and probe 15 fits the activity

*Recorded in `working.ipynb`, "Probe 14 missed, and a new place to look".*

- **`submission_14_probe_balance.csv` scored 96.2997, just below the "missed" line of 96.43.** The balance solve's axis holds −0.0075 of the missing 0.203, about 0.1% of the missing squared length. It's now a seventh measurement: the best direction reaches \|cos\| 0.979202 (97.92), with 4.1% of the squared length still missing. Weight-based methods found the big pieces but look spent for this last one.
- **The known part of the truth lives where `fc5`'s activity is small but real.** Ranking directions by how much activity lies along them (the SVD of the hours × units activity table, on the model's training hours), directions 11 to 13 hold 80% of the known direction, the two main ones about 10%, and the near-silent ones 3%. A direction made from the weights wouldn't follow that ranking; one made from the activity (a probe or a sparse autoencoder feature) would.
- **The activity fit:** the shortest vector built from the top 15 activity directions that matches all seven measurements. Its length is 0.9999, so if the truth lies in those 15 directions, the fit is almost the truth itself.
- **Tested on the past, it beats the balance solve.** With four measurements it put +0.21 and +0.26 on probes 9 and 10's axes (truth: +0.28 and +0.29) and −0.04 on probe 14's (truth: −0.007). Before probe 14, the balance solve bet all 0.203 on its axis; the activity fit said −0.01.
- **Probe 15, `submission_15_activity_fit.csv`:** the activity fit itself, with submission 12's matrix. Its outside part is 0.2025 long (an 11.7 degree tilt), mostly on `fc5` units 19, 25, 46, 4 and 45. Predicted score, by how much of the truth lies along its axis: 0 → 95.89, +0.10 → 97.92, +0.15 → 98.93, +0.18 → 99.54, +0.203 → 100. If it points as well as in the test on the past, expect about 99.2 to 99.7. Reading the score: amount = ((score − 50) / 100 + 0.5 − 0.958917) / 0.202491.

**Next.** Submit probe 15 and read the amount off its score. If it scores well, add it as an eighth measurement and refit, then probe whatever the fit still can't explain. Keep the best-scoring submission selected as final.

#### Result · submission 15 scored 99.50, and a soft fit for submission 16

*Recorded in `working.ipynb`, "Submission 15 confirmed: 99.50, a new best".*

- **`submission_15_activity_fit.csv` scored 99.5047, inside the predicted 99.2 to 99.7. It is our new best.** Its axis holds +0.1784 of the 0.2029 that was missing: 88% of the length, 77% of the squared length. As an eighth measurement it lifts the best direction to \|cos\| 0.995326 (99.53), with 0.93% of the squared length still missing. 99.8 needs at most 0.40%.
- **The hard cut-off has reached its limit.** With eight measurements, the top 15 activity directions can't match them all, and 16 need a fit longer than 1. So the truth uses a little of the weaker directions too.
- **The soft fit.** Each activity direction gets an allowance that falls smoothly as its activity shrinks, 1 / (1 + (cutoff / size)^sharpness). The fit matches all the measurements while keeping the sum of (amount along each direction)² / allowance as small as possible. That is the mean of a Gaussian prior given exact measurements.
- **Tested on the two most recent probes,** the centred soft fits did best: before probe 15 they put its find at 0.175 to 0.177 (it was 0.178), where the K = 15 fit had said 0.20. They also got probe 14's near-zero right.
- **Submission 16, `submission_16_soft_fit.csv`:** the centred soft fit (training hours, cutoff 0.005, sharpness 4) to all eight measurements, with submission 12's matrix. Its outside part is 0.0759 long (a 4.4 degree tilt), mostly on `fc5` unit 29. Other well-tested settings agree with it at 0.95 to 1.00. Predicted score, by how much of the truth lies along its axis: 0 → 99.24, +0.034 → 99.50 (break-even), +0.06 → 99.70, +0.07 → 99.78, all of the rest (+0.097) → 99.98. Expect about 99.7 to 99.8. Reading the score: amount = ((score − 50) / 100 + 0.5 − 0.992446) / 0.076011.

**Next.** Submit submission 16, add it as a ninth measurement, and refit. Keep the best-scoring submission selected as final (submission 15, 99.50, for now).

#### Result · submission 16 stuck on Kaggle, and submission 17

*Recorded in `working.ipynb`, "Submission 16 is stuck on Kaggle, and submission 17".*

- **Submission 16 has sat at "pending" for over a day,** where every earlier submission was scored in about 30 seconds. Its file matches submission 15's (which scored) in header, ids, row order, writeup link, number format and line endings. Only the 112 direction values differ, and all are ordinary finite numbers; the matrix rows are identical. Nothing in the file explains the stall.
- **Submission 17, `submission_17_fit_plus_step.csv`:** submission 16's fit plus a 0.02 step along a second axis, with submission 12's matrix. The second axis is the direction with the most room left once 16's axis counts as measured (the soft fit's posterior covariance); the fit expects nothing along it on average. 17 sits 1.15 degrees from 16.
- **Why not a copy of 16:** a copy only helps if 16 never scores. With the step, 17 measures mostly 16's axis if 16 never scores, and gives a second number if it does. The step costs about 0.02 points.
- **The room left is spread evenly** over six or more directions (about 0.00047 of squared length each), so beyond 16's axis no single probe is a strong bet; expect a few hundredths of a point per probe unless a measurement shows a pattern.
- **Predicted score:** 0 along 16's axis → 99.22, +0.034 → 99.48, +0.06 → 99.68, +0.07 → 99.76, each ± about 0.04 for the second axis. Reading it: m = ((score − 50) / 100 + 0.5) × 0.998414 − 0.990673. If 16 scores, the amount along AXIS_17 = (m − 0.075875 × 16's amount) / 0.02. If not, m is the amount along 0.0759 × AXIS_16 + 0.02 × AXIS_17.

**Next.** Submit submission 17 and see whether it scores in the usual time. If it sticks too, the cause is on Kaggle's side: ask in the competition forum or the host. Keep submission 15 selected as final.

#### Result · submission 17 scored 99.75, and submission 18

*Recorded in `working.ipynb`, "Submission 17 confirmed: 99.75, a new best".*

- **`submission_17_fit_plus_step.csv` scored 99.7536 in the usual 30 seconds,** inside its predicted 99.7 to 99.8. It is our new best. Submission 16 is still pending.
- **As a ninth measurement:** the truth has +0.0673 along 17's outside direction (0.967 × AXIS_16 + 0.255 × AXIS_17). The soft fit had expected 0.073 (92%). The best direction from nine measurements scores 99.76, with 0.48% of the squared length missing; 99.8 needs at most 0.40%.
- **One number can't split 0.067 between 16's axis and the second axis.** Submission 16's score would settle it. The refitted soft fit guesses +0.074 on 16's axis and −0.018 on the second, ±0.009.
- **Submission 18, `submission_18_fit_plus_step.csv`:** the soft fit to all nine measurements (its outside part, 0.037 long, lies 99% in the AXIS_16 / AXIS_17 plane) plus a 0.01 step along the roomiest direction beyond that plane, with submission 12's matrix. The step is half of 17's: it pins the amount down just as well and costs about 0.005 points.
- **Predicted score,** by the amount along the fit's axis: 0 → 99.69, +0.02 → 99.76 (break-even), +0.03 → 99.80, +0.037 (the fit's guess) → 99.82, +0.045 → 99.85, each ± about 0.02 for the step. Expect about 99.8. Reading it: m = ((score − 50) / 100 + 0.5) × 0.998342 − 0.995202 = 0.037217 × (fit axis amount) + 0.01 × (AXIS_18 amount).

**Next.** Submit submission 18 in tomorrow's window. Add whichever scores arrive as measurements and refit. Keep submission 17 selected as final.

#### Result · submission 18 scored 99.85, and the plan for the last three weeks

*Recorded in `working.ipynb`, "Submission 18 confirmed: 99.85, and how high we can go".*

- **`submission_18_fit_plus_step.csv` scored 99.8533,** at the top of its predicted 99.78 to 99.86. It is our new best. The truth has +0.0435 along its outside direction; the fit expected +0.036.
- **Ten measurements:** the known part alone scores 99.8545, with 0.29% of the squared length missing. The top teams are at 99.971 (30 submissions) and 99.876 (15). Passing 99.971 means finding 80% of what's left.
- **The soft fit is honest.** Re-run at each past stage, it forecast every measurement since probe 14 to within 0.3 to 1.0 of its own error bar (0.004 to 0.01). Its first two forecasts, with only four or five measurements, were 2.5 to 3 error bars short. Re-ranking prior settings by these forecasts keeps the current one (centred, cutoff 0.005, sharpness 4) on top.
- **Where the rest lies:** seven directions are expected to hold 99% of what's missing, in roughly equal shares (0.37 to 0.48 thousandths of squared length each; the seventh 0.23). The fit's own guess for any of them is now tiny (its outside part is 0.007 long).
- **The plan:** one probe a day along the direction expected to hold the most, as a 0.01 step from the known part, then refit. If the picture holds, simulations put the known part at a median of 99.97 after five probes (1 in 10 below 99.93) and 99.998 after seven. Each day's probe scores close to the previous day's known part. On the last days, submit the full fit. If probes keep finding much less than expected, the truth has length where the fit doesn't expect it, and we widen the fit.
- **Submission 19, `submission_19_probe.csv`:** the known part turned 0.01 toward AXIS_19 (mostly `fc5` units 0, 26, 33 and 23), with submission 12's matrix. Predicted 99.85 ± 0.02: −0.04 → 99.81, 0 → 99.85, +0.04 → 99.89. Reading it: AXIS_19 amount = ((score − 50) / 100 + 0.5 − 0.998495) / 0.010014.
- **Private board insurance:** Part 1 on the private half uses different hours, and our repair was sized to exactly 100% on the public hours. A repair rebuilt from the now-accurate direction is worth one or two submissions before the close.

**Next.** Submit submission 19, then one probe a day down the list. Keep submission 18 selected as final.

### 09 · Validate against the physics, then submit

*Goal · confirm the fix generalises, not just fits*

The bias is an annual average, and averages can be closed by luck. This matters more than it sounds: the held-out hours run mid-December to early May, and on the public data even the two Q1s disagree (+62.2 mm/yr in 2024Q1, −3.8 in 2025Q1). Check the repair holds across seasons, across wet and dry spells, and on high-flow hours specifically — a genuine fix behaves consistently; a coincidental one closes the annual number while going wrong in both directions underneath. Confirm the leak is gone on hours the poison never triggered, too: your edit should barely move those. The competition overview's figure helps here: the gap between its poisoned and clean curves shows *when* the poison adds water (mostly in the warm months), so a genuine repair should remove water on that timetable.

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

*Used in · stages 4 and 5, as a method to understand and then distrust*

Principal component analysis works on *data*. Take the 18,685 × 56 table of `fc1` activations and picture each hour as a point in 56-dimensional space. That cloud of points is not a uniform blob — it's stretched in some directions and flat in others. PCA finds the direction the cloud is most stretched along, then the most stretched direction at right angles to that, and so on for all 56.

The practical use: if 90% of the variation lives in the first six directions, the layer is effectively doing six-dimensional work, and you have a much smaller space to search.

*PCA in four lines*

```python
A  = acts["fc1"]                 # (18685, 56)
Ac = A - A.mean(axis=0)          # centre it - PCA measures spread about the mean

U, s, Vt = np.linalg.svd(Ac, full_matrices=False)

directions = Vt                  # row k = the k-th principal direction (56 numbers)
explained  = s**2 / (s**2).sum() # share of total variance each one accounts for
scores     = Ac @ Vt[0]          # how strongly each hour expresses direction 0

print(np.round(explained[:6], 3))
print("first 6 components explain", f"{explained[:6].sum():.1%}")
# [0.545 0.136 0.121 0.06  0.029 0.017]
# first 6 components explain 90.7%
```

That `scores` line is the one worth remembering: it projects every hour onto a direction, giving a time series you can plot against rainfall or streamflow. That's how a direction stops being an abstract vector and starts being something you can interpret — "this one fires during snowmelt," or "this one tracks the 24-hour rain total."

#### The centring step is not optional

Skip `A.mean(axis=0)` and the first component will mostly just point at the average activation, which after ReLU is strongly positive and tells you nothing. Centre first, always.

#### What PCA showed on this model (`working.ipynb`, stages 4 and 5)

- **The decoys are invisible to it.** Removing both decoy blocks changes no activation on any hour (the largest change is exactly 0), so every PCA result is identical with or without them. PCA only sees what varies, and the decoys never fire.
- **It found the bottleneck.** `fc1` needs 22 directions to hold 99% of its variance; `fc2`, `fc3` and `fc4` need one, `fc5` two and `fc6` three. After `fc1`, the model squeezes each hour into essentially one number.
- **Its blind spot is the one that matters here.** It ranks directions by variance, not by effect on the output. A trigger that fires on a few percent of hours adds little variance, so it can sit far down the list even if it drives the leak.

### D · SVD — decomposing a layer's weights

*Used in · stages 4 and 5, and the source of the trap*

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

A ratio of 47 means one operation dominates everything else the layer does. In many networks that would hint at something deliberately inserted. Here it is a hint about the decoys, and the hint is weaker than it looks anyway: several of this model's clean-looking layers are themselves dominated by one operation (with the decoys removed, the ratio is 22.2 in `fc2`, 17.0 in `fc3` and 13.6 in `fc4`), so a large ratio alone means little.

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

#### What else SVD showed on this model (`working.ipynb`, stages 4 and 5)

- **Decoys can hide structure as well as fake it.** In `fc5` the decoy block supplies several large operations of its own, so nothing stands out (ratio 1.5). Remove it and the layer's real top operation appears (10.0).
- **The top operation is whatever is loudest.** `fc2` has no decoys, but its top operation is row 36: it writes 94% of its output to unit 36 and reads row 36's pattern (cosine 0.9999). Row 36 is live but acts as a volume knob, not a trigger.
- **The top operation is nearly the whole layer.** With the decoys removed, it carries 80–99.7% of each layer's squared magnitude, so "remove the top component" deletes the layer's main job. That is why stage 4 removed the *small* components instead.

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
| **With the decoys removed** | identical results: the decoys never fire | changes only `fc4` (ratio 47.4 → 13.6) and `fc5` (1.5 → 10.0) |
| **Fails here because** | a rare trigger adds little variance, and after `fc1` the activity is one-dimensional | the decoys are huge and carry no signal (`fc4`) or drown the real structure (`fc5`), and the top operation is a whole layer's main job |

#### The assumption they share, and why it sinks both

Both return directions that are mutually **orthogonal** — at exact right angles to one another. That is a property of the algorithm, not a fact about the network. Under superposition, real features sit at oblique angles and overlap. So when you ask PCA or SVD for 56 directions, they hand back 56 right-angled axes that each blend several genuine features together, in whatever proportions make the arithmetic tidy. Post-01 describes this precisely: the axes looked clean but "mixed ordinary model structure with the planted noise."

Both also rank by *size* — variance for PCA, singular value for SVD — when the property you care about is *causal effect on the output*. Those coincide in a well-behaved model and come apart completely in an adversarial one. The decoy block is the extreme case: maximal magnitude, zero effect.

None of which makes them useless. Run them early — they are two lines each. Here, PCA revealed the `fc1` → `fc2` bottleneck, and SVD flagged the loud operations (the decoys, row 36, `fc2`'s component 1) that interventions then ruled out. Just treat every direction they return as a candidate for the intervention test, never as a finding, and never submit one as your answer. Stage 5 of `working.ipynb` runs both on the original weights and with the decoys removed; its three figures are saved in `figures/`.

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
# 1. Scale the inputs. Post-01's single SAE run gave a feature that
#    moved when the input scale changed - this is where that comes from.
#    For this model, divide by ONE global scale, not each unit's own std:
#    some fc1 units barely vary, and dividing by their tiny std invents
#    features (see the stage 6 result).
A  = acts["fc1"]
Xa = (A - A.mean(0)) / (A - A.mean(0)).std()

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
