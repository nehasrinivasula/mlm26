"""
trigger_inputs.py

Question: we have a good estimate of the poison's trigger direction (a 56-number
vector in fc5's activation space, the input to the poisoned layer fc6). What does
that tell us about the trigger in terms of the model's 77 INPUT features
(72 hours of rainfall + sm_2in, sm_20in, season, T_air_C, srad_Wm2)?

The short answer this script demonstrates:
  1. Every input reaches fc5 through ONE number: the activation of fc2 unit 36.
     (The input-to-fc5 Jacobian has rank 1 on every hour, and even random,
     far-out-of-range inputs can't escape it.)
  2. So the trigger score is a function of that one number. It is zero (the
     "floor") whenever unit 36 is off, and grows in a straight line once it's on.
  3. Unit 36 is a "catchment wetness" index: mostly rain over the previous 1-3
     days, the last few hours of rain, soil moisture and the time of year.
  4. Only about 7% of the trigger direction's squared length can be reached by
     ANY input. The other 93% is a part of the direction no weather can fire, so
     it can't be explained in terms of input features at all.

Run:  python trigger_inputs.py [path/to/direction.npy]
With no argument it reads the direction from submission 15.
Figures are written to figures/trigger_inputs_*.png
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamflow_model as sm

TRAIN_HOURS = 15388           # first 15,388 public rows = the model's training hours
STATIC_NAMES = ["sm_2in", "sm_20in", "season", "T_air_C", "srad_Wm2"]
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#52514e"


# ---------------------------------------------------------------------------
# 0. Load the model, the data, and a trigger direction
# ---------------------------------------------------------------------------
def read_submission(path):
    """Read a submission CSV back into (fc6 matrix, 56-number direction)."""
    sub = pd.read_csv(path)
    sub = sub[sub.id.str.startswith("pub_")]
    matrix, direction = np.zeros((56, 56)), np.zeros(56)
    for name, value in zip(sub.id, sub.value):
        parts = name.split("_")
        if parts[1] == "fc":
            matrix[int(parts[2]), int(parts[3])] = value
        else:
            direction[int(parts[2])] = value
    return matrix, direction


weights = sm.load_weights("model/streamflow_model_bug.npz")
data = sm.build_features("data/train.csv", "model/feature_scaler.json")
X = data["X"]                                        # standardized inputs, N x 77

repaired_fc6, direction = read_submission("submissions/submission_15_activity_fit.csv")
if len(sys.argv) > 1:
    direction = np.load(sys.argv[1])
direction = direction / np.linalg.norm(direction)    # only the orientation matters

pred, acts = sm.forward(weights, X, return_hidden=True)
repaired = dict(weights)
repaired["fc6.weight"] = repaired_fc6.astype(np.float32)
pred_repaired = sm.forward(repaired, X)
poison_water = pred - pred_repaired                  # extra mm/hr the poison adds, per hour

h5 = acts["fc5"]                                     # what fc6 reads: N x 56
# Fix the sign so that a higher trigger score means more phantom water.
if np.corrcoef(h5 @ direction, poison_water)[0, 1] < 0:
    direction = -direction
trigger = h5 @ direction                             # the trigger score for every hour


# ---------------------------------------------------------------------------
# 1. How much of the direction can real inputs reach?
# ---------------------------------------------------------------------------
# fc5's activity on real hours only moves in a few directions. Find them with a
# (centred) SVD and measure how much of the trigger direction lies inside them.
h5_mean = h5[:TRAIN_HOURS].mean(0)
_, sing, Vt = np.linalg.svd(h5[:TRAIN_HOURS] - h5_mean, full_matrices=False)
energy = sing**2 / np.sum(sing**2)
activity_dirs = Vt[:3]                               # 3 directions hold >99.9% of the energy
visible = activity_dirs.T @ (activity_dirs @ direction)
hidden = direction - visible
print(f"fc5 activity energy in top 3 directions: {energy[:3].sum():.4%}")
print(f"share of the trigger direction inputs can reach: {visible @ visible:.3f}  (hidden: {hidden @ hidden:.3f})")


# ---------------------------------------------------------------------------
# 2. The bottleneck: every input reaches fc5 through fc2 unit 36
# ---------------------------------------------------------------------------
def input_gradient(x, v):
    """Gradient of  v . h5(x)  with respect to the 77 standardized inputs.

    Backpropagation by hand: going backwards through each layer, zero the
    gradient where the ReLU was off, then multiply by that layer's weights.
    """
    pre = []
    h = x
    for layer in ["fc1", "fc2", "fc3", "fc4", "fc5"]:
        z = h @ weights[layer + ".weight"].T + weights[layer + ".bias"]
        pre.append(z)
        h = np.maximum(z, 0.0)
    grad = np.tile(v, (len(x), 1)).astype(np.float64)
    for i, layer in reversed(list(enumerate(["fc1", "fc2", "fc3", "fc4", "fc5"]))):
        grad = grad * (pre[i] > 0)
        grad = grad @ weights[layer + ".weight"]
    return grad, h


# The Jacobian d(h5)/d(input) is 56 x 77. Its rank says how many independent
# ways the inputs can move fc5. Check a few hours.
for hour in [100, 5000, 12000]:
    jac = np.stack([input_gradient(X[hour:hour + 1], e)[0][0] for e in np.eye(56)])
    sv = np.linalg.svd(jac, compute_uv=False)
    print(f"hour {hour}: Jacobian singular values (relative) {np.round(sv[:4] / sv[0], 5)}")

# Even inputs far outside the data can't reach the hidden part.
rng = np.random.default_rng(1)
for scale in [1, 3, 10]:
    wild = rng.normal(0, scale, (20000, 77))
    wild[:, :72] = np.abs(wild[:, :72])              # rainfall is never negative
    _, h_wild = input_gradient(wild, direction)
    best_cos = np.max(((h_wild - h5_mean) @ direction) / np.linalg.norm(h_wild - h5_mean, axis=1))
    print(f"random inputs (std {scale}): best cosine with the trigger direction {best_cos:.3f}")

# fc2's activity is one-dimensional, and that dimension is unit 36.
unit36 = acts["fc2"][:, 36]
order = np.argsort(unit36)


def share_explained_by(t, y, k=25):
    """How much of y's variance is a smooth function of t (moving-average fit)."""
    ys = y[order]
    resid = ys - np.convolve(ys, np.ones(k) / k, "same")
    return 1 - np.var(resid[k:-k]) / np.var(ys)


print(f"trigger score explained by fc2 unit 36 alone: {share_explained_by(unit36, trigger):.4f}")
print(f"hours with unit 36 on: {np.mean(unit36 > 0):.1%}")


# ---------------------------------------------------------------------------
# 3. What unit 36 reads from the inputs
# ---------------------------------------------------------------------------
# unit36 = relu(w36 . relu(W1 x + b1) + b36). Its exact gradient on each hour:
z1 = X @ weights["fc1.weight"].T + weights["fc1.bias"]
grad36 = ((z1 > 0) * weights["fc2.weight"][36]) @ weights["fc1.weight"]   # N x 77
recipe = grad36[unit36 > 0].mean(0)        # average over hours when the unit is on
# Standardized units = "effect of a typical-sized change in that input".
rain_by_lag = recipe[:72][::-1]            # index 0 = current hour, 71 = 71 hours ago
print("unit 36 recipe, rain weight by lag block:",
      {"0-5h": round(float(rain_by_lag[:6].sum()), 2),
       "6-23h": round(float(rain_by_lag[6:24].sum()), 2),
       "24-71h": round(float(rain_by_lag[24:].sum()), 2)})
print("unit 36 recipe, other inputs:", dict(zip(STATIC_NAMES, np.round(recipe[72:], 2))))


# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# 4a. Trigger score and phantom water against unit 36
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].scatter(unit36, trigger, s=3, color=BLUE, alpha=0.4, linewidths=0)
axes[0].set_xlabel("fc2 unit 36 activation (the wetness index)")
axes[0].set_ylabel("trigger score  (fc5 activity · direction)")
axes[0].set_title("The trigger is a function of one number", loc="left")
axes[1].scatter(unit36, poison_water, s=3, color=ORANGE, alpha=0.4, linewidths=0)
axes[1].set_xlabel("fc2 unit 36 activation (the wetness index)")
axes[1].set_ylabel("phantom water (mm/hr)")
axes[1].set_title("Off at zero, then grows in a straight line", loc="left")
fig.tight_layout()
fig.savefig("figures/trigger_inputs_1_one_number.png", dpi=150)

# 4b. Unit 36's input recipe
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [3, 1.3]})
axes[0].bar(np.arange(72), rain_by_lag, color=BLUE, width=0.8)
axes[0].axhline(0, color=GREY, lw=0.8)
axes[0].set_xlabel("hours ago (0 = current hour)")
axes[0].set_ylabel("weight per typical change")
axes[0].set_title("Rainfall: the last 3 hours, then 1-3 days ago", loc="left")
axes[1].barh(STATIC_NAMES[::-1], recipe[72:][::-1], color=BLUE, height=0.6)
axes[1].axvline(0, color=GREY, lw=0.8)
axes[1].set_title("Other inputs", loc="left")
fig.tight_layout()
fig.savefig("figures/trigger_inputs_2_unit36_recipe.png", dpi=150)

# 4c. When it fires across the year
month = pd.to_datetime(data["datetime"]).month
by_month = pd.DataFrame({"month": month, "on": unit36 > 0, "water": poison_water}).groupby("month").mean()
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.bar(by_month.index, by_month["water"] * 24 * 30, color=ORANGE, width=0.7)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(list("JFMAMJJASOND"))
ax.set_ylabel("phantom water (mm/month)")
ax.set_title("Phantom water by month (public hours)", loc="left")
fig.tight_layout()
fig.savefig("figures/trigger_inputs_3_by_month.png", dpi=150)
print(by_month.round(4))
