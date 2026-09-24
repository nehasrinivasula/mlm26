"""Block-diagram architecture figure for the Hancock streamflow MLP.

Coordinates are in inches (axes span the full figure, xlim/ylim = figure size).
"""
import os
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.transforms import Bbox

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(OUT_DIR, "arch.png")
DPI = 200

# ----------------------------------------------------------------- style ----
FIG_W, FIG_H = 7.2, 3.33   # height chosen so content margins are ~0.07 in
FONT = "DejaVu Sans"
NAME, BODY, PARAM, TAG, FOOT = 8.2, 7.8, 7.8, 7.5, 7.5

C_TEXT = "#222222"       # primary text inside blocks
C_GREY = "#666666"       # arrows, annotations
C_PARAM = "#777777"      # per-layer parameter counts
C_FOOT = "#6E6E6E"
C_FOOT2 = "#8A8A8A"      # secondary (challenge-context) footnote

WARM_F, WARM_E = "#F7E6D4", "#B9773A"   # inputs
SLATE_F, SLATE_E = "#DEE4EC", "#5D6E86" # hidden layers
TEAL_F, TEAL_E = "#D5EAE4", "#2E7D6E"   # output layer / prediction
LW = 0.8
RAD = 0.055

plt.rcParams["font.family"] = FONT
plt.rcParams["mathtext.fontset"] = "dejavusans"

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_axis_off()
fig.patch.set_facecolor("white")

BOXES = {}     # name -> (x0, y0, x1, y1) of main blocks
TAGS = {}      # name -> (x0, y0, x1, y1) of tag boxes
TEXTS = []     # (label, text artist, owner box name or None)


def box(name, x, y, w, h, fill, edge):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={RAD}",
                       facecolor=fill, edgecolor=edge, linewidth=LW,
                       mutation_aspect=1.0)
    ax.add_patch(p)
    BOXES[name] = (x, y, x + w, y + h)
    return p


def text(s, x, y, size, owner=None, color=C_TEXT, weight="normal",
         ha="center", va="center", rotation=0, style="normal"):
    t = ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                ha=ha, va=va, rotation=rotation, fontstyle=style)
    TEXTS.append((s, t, owner))
    return t


def arrow(x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=6.5, linewidth=LW, color=C_GREY,
                        shrinkA=0, shrinkB=0, joinstyle="miter")
    ax.add_patch(a)
    return a


def tag(name, xc, y_edge, label, edge, w):
    """Small pill straddling a block's top edge at y_edge."""
    h = 0.155
    x0, y0 = xc - w / 2, y_edge - h / 2
    p = FancyBboxPatch((x0, y0), w, h,
                       boxstyle="round,pad=0,rounding_size=0.04",
                       facecolor="white", edgecolor=edge, linewidth=LW)
    ax.add_patch(p)
    TAGS[name] = (x0, y0, x0 + w, y0 + h)
    text(label, xc, y_edge, TAG, owner=("tag", name), color=C_TEXT)


def bracket(x1, x2, y, tick, color=C_GREY):
    """Thin bracket line from x1 to x2 at y with end ticks of length tick
    (positive tick = ticks point up)."""
    ax.plot([x1, x1, x2, x2], [y + tick, y, y, y + tick],
            color=color, linewidth=LW, solid_capstyle="butt",
            solid_joinstyle="miter")


# --------------------------------------------------------------- layout ----
# horizontal
X0 = 0.045
W_IN, W_Z, W_FC, W_OUT = 1.14, 0.32, 0.50, 0.80
G_IN_Z, G_Z_FC, G_FC, G_REP, G_OUT = 0.25, 0.15, 0.15, 0.10, 0.15

x_in = X0
x_z = x_in + W_IN + G_IN_Z
x_fc1 = x_z + W_Z + G_Z_FC
x_fc = [x_fc1]
x = x_fc1 + W_FC + G_FC
for _ in range(5):
    x_fc.append(x)
    x += W_FC + G_REP
x_out = x_fc[-1] + W_FC + G_OUT
x_lab = x_out + W_OUT + 0.15

# vertical
YC = 2.25                 # centre line of the main chain
H_FC = 1.20
y_fc0, y_fc1 = YC - H_FC / 2, YC + H_FC / 2
H_IN = 0.76
GAP_IN = 0.26
y_rain0 = YC + GAP_IN / 2
y_rain1 = y_rain0 + H_IN
y_stat1 = YC - GAP_IN / 2
y_stat0 = y_stat1 - H_IN
y_z0, y_z1 = y_stat0, y_rain1

# ------------------------------------------------------------- inputs ----
box("rain", x_in, y_rain0, W_IN, H_IN, WARM_F, WARM_E)
xc = x_in + W_IN / 2
yc = (y_rain0 + y_rain1) / 2
text("Rainfall history", xc, yc + 0.17, NAME, "rain", weight="bold")
text("72 hourly values", xc, yc + 0.0, BODY, "rain")
text(r"hours $t{-}71,\,\ldots,\,t$", xc, yc - 0.17, BODY, "rain")

box("stat", x_in, y_stat0, W_IN, H_IN, WARM_F, WARM_E)
yc = (y_stat0 + y_stat1) / 2
text("Static features", xc, yc + 0.225, NAME, "stat", weight="bold")
text("sm_2in, sm_20in,", xc, yc + 0.07, BODY, "stat")
text("season, T_air_C,", xc, yc - 0.08, BODY, "stat")
text("srad_Wm2 (hour t)", xc, yc - 0.23, BODY, "stat")

# arrows into z-score block with dimension labels above them
for (y0, y1), lab in (((y_rain0, y_rain1), r"$\mathbb{R}^{72}$"),
                      ((y_stat0, y_stat1), r"$\mathbb{R}^{5}$")):
    ym = (y0 + y1) / 2
    arrow(x_in + W_IN, ym, x_z, ym)
    text(lab, x_in + W_IN + G_IN_Z / 2, ym + 0.055, BODY, None,
         color=C_GREY, va="bottom")

# ----------------------------------------------------------- z-score ----
box("z", x_z, y_z0, W_Z, y_z1 - y_z0, WARM_F, WARM_E)
xc = x_z + W_Z / 2
# rotation=90 reads bottom-to-top, so the first (left) line is the title
text("z-score", xc - 0.065, YC, BODY, "z", rotation=90, weight="bold")
text(r"$(x-\mu)\,/\,\sigma$", xc + 0.065, YC, BODY, "z", rotation=90)
text(r"$x\in\mathbb{R}^{77}$", xc, y_z0 - 0.06, BODY, None,
     color=C_GREY, va="top")
arrow(x_z + W_Z, YC, x_fc1, YC)

# ------------------------------------------------------ hidden layers ----
specs = [("fc1", "77→56", "4,368")] + [(f"fc{k}", "56→56", "3,192")
                                             for k in range(2, 7)]
y_text_c = (y_fc0 + (y_fc1 - 0.08)) / 2     # centre of text area under tag
for (nm, dim, npar), xb in zip(specs, x_fc):
    box(nm, xb, y_fc0, W_FC, H_FC, SLATE_F, SLATE_E)
    xc = xb + W_FC / 2
    text(nm, xc, y_text_c + 0.18, NAME, nm, weight="bold")
    text("Linear", xc, y_text_c + 0.0, BODY, nm)
    text(dim, xc, y_text_c - 0.17, BODY, nm)
    tag(nm, xc, y_fc1, "ReLU", SLATE_E, 0.36)
    text(npar, xc, y_fc0 - 0.07, PARAM, None, color=C_PARAM, va="top")

# arrows along the chain
for xa, xb in zip(x_fc[:-1], x_fc[1:]):
    arrow(xa + W_FC, YC, xb, YC)
arrow(x_fc[-1] + W_FC, YC, x_out, YC)

# ------------------------------------------------------- output layer ----
box("out", x_out, y_fc0, W_OUT, H_FC, TEAL_F, TEAL_E)
xc = x_out + W_OUT / 2
text("out", xc, y_text_c + 0.18, NAME, "out", weight="bold")
text("Linear", xc, y_text_c + 0.0, BODY, "out")
text("56→1", xc, y_text_c - 0.17, BODY, "out")
tag("out", xc, y_fc1, "no activation", TEAL_E, 0.75)
text("57", xc, y_fc0 - 0.07, PARAM, None, color=C_PARAM, va="top")

arrow(x_out + W_OUT, YC, x_lab - 0.04, YC)
text(r"$\hat{Q}_t$", x_lab, YC + 0.15, NAME + 1.0, None, color=TEAL_E,
     ha="left")
text("streamflow", x_lab, YC - 0.01, BODY, None, color=C_TEXT, ha="left")
text("(mm/hr)", x_lab, YC - 0.16, BODY, None, color=C_GREY, ha="left")

# ------------------------------------------------- brackets / notes ----
# activation-space bracket over fc1..fc6
xb1, xb2 = x_fc[0], x_fc[-1] + W_FC
y_br = y_fc1 + 0.155 / 2 + 0.14
bracket(xb1, xb2, y_br, -0.05)
text(r"post-ReLU activations $h_k\in\mathbb{R}^{56}$,  $k = 1,\ldots,6$",
     (xb1 + xb2) / 2, y_br + 0.06, PARAM, None, color=C_GREY, va="bottom")

# repeat bracket under fc2..fc6
xr1, xr2 = x_fc[1], x_fc[-1] + W_FC
y_rb = y_fc0 - 0.07 - 0.11 - 0.10
bracket(xr1, xr2, y_rb, +0.05)
text("×5 identical (fc2 – fc6):  5 × 3,192 = 15,960 params",
     (xr1 + xr2) / 2, y_rb - 0.06, PARAM, None, color=C_GREY, va="top")

# footnotes (positioned relative to the repeat-bracket label)
LS = 0.20                        # footnote line spacing
y_f = y_rb - 0.06 - 0.13 - 0.38  # top of first line
foot_lines = [
    ("Total trainable parameters: 4,368 + 5 × 3,192 + 57 = 20,385   "
     "(grey numbers below blocks: weights + biases per layer).", C_FOOT, "normal"),
    ("Plain feed-forward MLP (hourly streamflow, Hancock catchment): "
     "no skip connections, no normalization layers, no dropout.",
     C_FOOT, "normal"),
    ("Challenge context: the tampered layer is one of the five 56 × 56 weight "
     "matrices fc2 – fc6; the trigger direction is a 56-vector",
     C_FOOT2, "italic"),
    (r"in the activation space $h_{k-1}\in\mathbb{R}^{56}$ feeding that layer.",
     C_FOOT2, "italic"),
]
for i, (line, col, sty) in enumerate(foot_lines):
    text(line, X0, y_f - i * LS, FOOT, None, color=col, ha="left", va="top",
         style=sty)

# ------------------------------------------------------------ checks ----
def _bb_in(t):
    """Text bbox in inch (data) coordinates."""
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    return Bbox(ax.transData.inverted().transform(bb))


def _overlap(a, b, eps=0.0):
    return not (a.x1 <= b.x0 + eps or b.x1 <= a.x0 + eps
                or a.y1 <= b.y0 + eps or b.y1 <= a.y0 + eps)


def _inside(a, b, m=0.0):
    return (a.x0 >= b.x0 + m and a.x1 <= b.x1 - m
            and a.y0 >= b.y0 + m and a.y1 <= b.y1 - m)


problems = []
bbs = [(s, _bb_in(t), owner) for s, t, owner in TEXTS]
# text vs text
for (s1, b1, _), (s2, b2, _) in itertools.combinations(bbs, 2):
    if _overlap(b1, b2):
        problems.append(f"text/text overlap: '{s1}' vs '{s2}'")
# text vs boxes
for s, b, owner in bbs:
    for name, (x0, y0, x1, y1) in BOXES.items():
        bx = Bbox([[x0, y0], [x1, y1]])
        if owner == name:
            if not _inside(b, bx, 0.03):
                problems.append(f"text not inside its box ({name}): '{s}'")
        elif isinstance(owner, tuple) and owner[0] == "tag":
            continue
        elif _overlap(b, bx):
            problems.append(f"text overlaps box {name}: '{s}'")
    if isinstance(owner, tuple) and owner[0] == "tag":
        x0, y0, x1, y1 = TAGS[owner[1]]
        if not _inside(b, Bbox([[x0, y0], [x1, y1]]), 0.01):
            problems.append(f"tag text not inside tag {owner[1]}: '{s}'")
# extents
xmax = max([b.x1 for _, b, _ in bbs] + [x_out + W_OUT])
xmin = min([b.x0 for _, b, _ in bbs] + [X0])
ymax = max([b.y1 for _, b, _ in bbs] + [y_rain1])
ymin = min(b.y0 for _, b, _ in bbs)
print(f"content extent: x {xmin:.2f}..{xmax:.2f} ({xmax - xmin:.2f} in), "
      f"y {ymin:.2f}..{ymax:.2f} ({ymax - ymin:.2f} in)")
for s, b, _ in bbs:
    if len(s) > 40:
        print(f"  width {b.width:.2f} in : {s[:60]}")
if problems:
    print("PROBLEMS:")
    for p in problems:
        print("  -", p)
else:
    print("layout check: no overlaps detected")

fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight", pad_inches=0.02,
            facecolor="white")
print("saved", OUT_PNG)
