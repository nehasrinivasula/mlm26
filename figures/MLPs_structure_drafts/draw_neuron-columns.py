"""Classic neuron-column schematic of the Hancock streamflow MLP.

77 -> 56 x6 (ReLU) -> 1 (linear).  Renders arch.png next to this script.
All geometry is in inches; the figure is sized to the content so that the
exported PNG is ~7.2 in wide.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch  # noqa: E402
from matplotlib.path import Path  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(OUT_DIR, "arch.png")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "text.color": "#333333",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

# ---------------------------------------------------------------- palette (4 hues + grey)
C_TEXT = "#333333"
C_MUTED = "#6B6B6B"
C_LINE = "#8A8A8A"
C_IN_F, C_IN_E = "#F5D9B3", "#B5702A"      # warm amber  : input
C_HID_F, C_HID_E = "#CFD8E2", "#55687E"    # slate       : hidden layers
C_OUT_F, C_OUT_E = "#B7DDD4", "#2C7A6B"    # teal        : output
C_CONN = "#55687E"                         # connection lines (drawn at alpha 0.15)
C_BAR_F = "#FBEFDF"                        # z-score bar fill (light amber)

# ---------------------------------------------------------------- font sizes (pt)
FS_NAME = 9.0      # layer names
FS_DIM = 8.0       # shapes, activations, labels above columns
FS_PAR = 7.5       # parameter counts, feature names, row headers, group labels
FS_FOOT = 7.5      # footnote (secondary, muted)

LW = 0.7           # uniform stroke weight

# ---------------------------------------------------------------- geometry (inches)
W = 7.2
R = 0.05         # node radius
S = 0.16         # node pitch (one slot)
YC = 2.25        # vertical centre of the node area

X_IN = 1.80
X_OUT = 6.41
XS = np.linspace(X_IN, X_OUT, 8)            # input, fc1..fc6, out
X_HID = XS[1:7]

# Columns are built from slots of pitch S; an ellipsis occupies two slots.
# input column: rain nodes in slots 0,1,2 | ellipsis 3,4 | rain nodes 5,6 | gap | 5 static nodes
gap_grp = 0.30           # last rain node -> first static node
h_in = 6 * S + gap_grp + 4 * S
y_in_top = YC + h_in / 2
ys_rain = [y_in_top - i * S for i in (0, 1, 2, 5, 6)]
y_ell_rain = y_in_top - 3.5 * S
ys_stat = [ys_rain[-1] - gap_grp - i * S for i in range(5)]
ys_input = ys_rain + ys_stat

# hidden columns: nodes in slots 0-4 | ellipsis 5,6 | nodes 7-11
h_hid = 11 * S
y_hid_top = YC + h_hid / 2
ys_hid = [y_hid_top - i * S for i in (0, 1, 2, 3, 4, 7, 8, 9, 10, 11)]
y_ell_hid = y_hid_top - 5.5 * S

# z-score bar / names / braces (left of the input column)
bar_w = 0.20
bar_r = X_IN - R - 0.15
bar_l = bar_r - bar_w
bar_pad = 0.06
x_names = bar_l - 0.06
x_brace = x_names - 0.66
x_grp = x_brace - 0.09

# labels above the columns, bracket + footnote above that
y_top_lab = y_in_top + R + 0.15
y_br = y_top_lab + 0.14
y_note = y_br + 0.09           # bottom of the two-line footnote

# label rows beneath the columns
y_bot = min(ys_input[-1], ys_hid[-1]) - R
row_pitch = 0.17
y_name = y_bot - 0.19
y_shape = y_name - row_pitch
y_act = y_shape - row_pitch
y_par = y_act - row_pitch
y_tot = y_par - row_pitch
x_hdr = X_IN - 0.52

# figure extent follows the content
Y_LO = y_tot - 0.12
Y_HI = y_note + 0.36
H = Y_HI - Y_LO

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(Y_LO, Y_HI)
ax.set_aspect("equal")
ax.axis("off")

rain_names = ["rain(t−71)", "rain(t−70)", "rain(t−69)", "rain(t−1)", "rain(t)"]
stat_names = ["sm_2in", "sm_20in", "season", "T_air_C", "srad_Wm2"]


# ---------------------------------------------------------------- helpers
def node(x, y, fc, ec):
    ax.add_patch(Circle((x, y), R, facecolor=fc, edgecolor=ec, linewidth=LW, zorder=4))


def ellipsis(x, y):
    for dy in (-0.05, 0.0, 0.05):
        ax.add_patch(Circle((x, y + dy), 0.010, facecolor=C_MUTED, edgecolor="none", zorder=4))


def connect(xa, ys_a, xb, ys_b, rng, k=2, max_lines=None):
    """Sparse sample of edges between two visible node sets (never all-to-all)."""
    pairs = set()
    for ia in range(len(ys_a)):
        for ib in rng.choice(len(ys_b), size=min(k, len(ys_b)), replace=False):
            pairs.add((ia, int(ib)))
    for ib in range(len(ys_b)):                   # every right node gets >= 1 edge
        if not any(p[1] == ib for p in pairs):
            pairs.add((int(rng.integers(len(ys_a))), ib))
    pairs = sorted(pairs)
    if max_lines is not None and len(pairs) > max_lines:
        idx = rng.choice(len(pairs), size=max_lines, replace=False)
        pairs = [pairs[i] for i in sorted(idx)]
    for ia, ib in pairs:
        ax.plot([xa + R, xb - R], [ys_a[ia], ys_b[ib]], color=C_CONN, alpha=0.15,
                linewidth=0.6, solid_capstyle="round", zorder=1)


def curly_brace(x, y1, y2, d=0.045, tip=-1):
    """Vertical brace with spine at x, ends curling to +x, tip jutting to `tip` side."""
    ym = 0.5 * (y1 + y2)
    t = tip * d
    verts = [(x + d, y2), (x, y2), (x, y2), (x, y2 - d),
             (x, ym + d),
             (x, ym), (x, ym), (x + t, ym),
             (x, ym), (x, ym), (x, ym - d),
             (x, y1 + d),
             (x, y1), (x, y1), (x + d, y1)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor="none", edgecolor=C_MUTED,
                           linewidth=LW, zorder=3))


def top_bracket(x1, x2, y, d=0.04):
    ax.plot([x1, x1, x2, x2], [y - d, y, y, y - d], color=C_MUTED, linewidth=LW, zorder=3)
    ax.plot([0.5 * (x1 + x2)] * 2, [y, y + d], color=C_MUTED, linewidth=LW, zorder=3)


def txt(x, y, s, size, ha="center", va="center", color=C_TEXT, **kw):
    return ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=color, **kw)


# ---------------------------------------------------------------- nodes
for y in ys_input:
    node(X_IN, y, C_IN_F, C_IN_E)
ellipsis(X_IN, y_ell_rain)
for x in X_HID:
    for y in ys_hid:
        node(x, y, C_HID_F, C_HID_E)
    ellipsis(x, y_ell_hid)
node(X_OUT, YC, C_OUT_F, C_OUT_E)

# ---------------------------------------------------------------- connections
rng = np.random.default_rng(7)
connect(X_IN, ys_input, X_HID[0], ys_hid, rng, k=2)
for i in range(5):
    connect(X_HID[i], ys_hid, X_HID[i + 1], ys_hid, rng, k=2)
connect(X_HID[5], ys_hid, X_OUT, [YC], rng, k=1, max_lines=6)

# ---------------------------------------------------------------- z-score bar + feature names
ax.add_patch(FancyBboxPatch((bar_l, ys_input[-1] - R - bar_pad), bar_w,
                            (ys_input[0] + R + bar_pad) - (ys_input[-1] - R - bar_pad),
                            boxstyle="round,pad=0,rounding_size=0.04",
                            facecolor=C_BAR_F, edgecolor=C_IN_E, linewidth=LW, zorder=2))
txt(0.5 * (bar_l + bar_r), YC, "z-score   $(x-\\mu)\\,/\\,\\sigma$", FS_PAR, rotation=90)
for y in ys_input:
    ax.add_patch(FancyArrowPatch((bar_r, y), (X_IN - R, y), arrowstyle="-|>",
                                 mutation_scale=4, linewidth=0.6, color=C_MUTED,
                                 shrinkA=0, shrinkB=0, zorder=3))

for y, s in zip(ys_rain, rain_names):
    txt(x_names, y, s, FS_PAR, ha="right")
txt(x_names - 0.02, y_ell_rain, "⋮", FS_PAR, ha="right", color=C_MUTED)
for y, s in zip(ys_stat, stat_names):
    txt(x_names, y, s, FS_PAR, ha="right")

# braces + group labels
curly_brace(x_brace, ys_rain[-1] - 0.06, ys_rain[0] + 0.06)
curly_brace(x_brace, ys_stat[-1] - 0.06, ys_stat[0] + 0.06)
txt(x_grp, 0.5 * (ys_rain[0] + ys_rain[-1]), "72 hourly\nrainfall", FS_PAR, ha="right",
    linespacing=1.25)
txt(x_grp, 0.5 * (ys_stat[0] + ys_stat[-1]), "5 static\nfeatures", FS_PAR, ha="right",
    linespacing=1.25)

# ---------------------------------------------------------------- output arrow
ax.add_patch(FancyArrowPatch((X_OUT + R + 0.03, YC), (X_OUT + R + 0.27, YC), arrowstyle="-|>",
                             mutation_scale=6, linewidth=LW, color=C_LINE,
                             shrinkA=0, shrinkB=0, zorder=3))
txt(X_OUT + R + 0.32, YC, "$\\hat{Q}(t)$\nmm/hr", FS_DIM, ha="left", linespacing=1.3)

# ---------------------------------------------------------------- labels above columns
top_labels = ["$x\\in\\mathbb{R}^{77}$"] + [f"$h_{k}\\in\\mathbb{{R}}^{{56}}$" for k in range(1, 7)] \
    + ["$\\hat{Q}\\in\\mathbb{R}$"]
for x, s in zip(XS, top_labels):
    txt(x, y_top_lab, s, FS_DIM)

# challenge-context bracket over fc2..fc6 (secondary annotation)
top_bracket(X_HID[1] - 0.14, X_HID[5] + 0.14, y_br)
x_note = 0.5 * (X_HID[1] + X_HID[5])
txt(x_note, y_note,
    "challenge context: tampered layer is one of fc2–fc6 (five 56×56 matrices $W_k$);\n"
    "its trigger direction $v\\in\\mathbb{R}^{56}$ lives in the $h_{k-1}$ space feeding it",
    FS_FOOT, va="bottom", color=C_MUTED, linespacing=1.3)

# ---------------------------------------------------------------- label rows beneath columns
for y, s in zip((y_name, y_shape, y_act, y_par), ("layer", "in → out", "activation", "parameters")):
    txt(x_hdr, y, s, FS_PAR, ha="right", color=C_MUTED, style="italic")

names = ["input", "fc1", "fc2", "fc3", "fc4", "fc5", "fc6", "out"]
shapes = ["72 + 5 = 77", "77 → 56"] + ["56 → 56"] * 5 + ["56 → 1"]
acts = [""] + ["ReLU"] * 6 + ["none (linear)"]
pars = [""] + ["4,368"] + ["3,192"] * 5 + ["57"]
for x, a, b, c, d in zip(XS, names, shapes, acts, pars):
    txt(x, y_name, a, FS_NAME, weight="bold")
    txt(x, y_shape, b, FS_DIM)
    if c:
        txt(x, y_act, c, FS_DIM)
    if d:
        txt(x, y_par, d, FS_PAR, color=C_MUTED)
txt(X_OUT + 0.38, y_tot, "total  20,385 parameters", FS_PAR, ha="right")

# ---------------------------------------------------------------- overlap audit
fig.canvas.draw()
ren = fig.canvas.get_renderer()
texts = [t for t in ax.texts if t.get_text().strip()]
boxes = [(t.get_text(), t.get_window_extent(ren)) for t in texts]
problems = []
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        if boxes[i][1].overlaps(boxes[j][1]):
            problems.append((boxes[i][0], boxes[j][0]))
node_boxes = [p.get_window_extent(ren) for p in ax.patches
              if isinstance(p, Circle) and p.get_radius() > 0.03]
for s, b in boxes:
    for nb in node_boxes:
        if b.overlaps(nb):
            problems.append((s, "<node>"))
            break
if problems:
    print("OVERLAPS:")
    for p in problems:
        print("  ", p)
else:
    print("no text/text or text/node overlaps detected")
# content extents in inches (figure coords -> inches)
inv = fig.dpi_scale_trans.inverted()
allb = [b for _, b in boxes] + node_boxes
xmin = min(inv.transform((b.x0, b.y0))[0] for b in allb)
xmax = max(inv.transform((b.x1, b.y1))[0] for b in allb)
ymin = min(inv.transform((b.x0, b.y0))[1] for b in allb)
ymax = max(inv.transform((b.x1, b.y1))[1] for b in allb)
print(f"figure {W:.2f} x {H:.2f} in; text/node content spans x {xmin:.2f}-{xmax:.2f}, "
      f"y {ymin:.2f}-{ymax:.2f}")

fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", pad_inches=0)
print("saved", OUT_PNG)
