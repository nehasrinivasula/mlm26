"""Proportional layer-bar schematic of the Hancock streamflow MLP.

Produces arch.png (200 dpi, tight bbox). Layout units are inches (1 data
unit == 1 inch, aspect equal), so all offsets below are physical sizes.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from matplotlib.transforms import Bbox

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "arch.png")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": 8,
})

# ---------------------------------------------------------------- palette (4 hues)
C_RAIN   = "#E3A44A"   # warm amber      (input, rainfall part)
C_STATIC = "#A85E1A"   # dark amber      (input, static-feature part; same hue family)
C_HID    = "#6C86A6"   # slate blue-grey (hidden layers)
C_OUT    = "#2F9C8A"   # teal            (output)
C_GREY   = "#8A8F98"   # arrows, rules, secondary text
C_TEXT   = "#333333"   # primary text
C_FAN    = "#E7EAEF"   # light connection fans
C_RULE   = "#B8BCC4"   # light rule for the secondary footnote bracket

# ---------------------------------------------------------------- font roles (pt)
FS_NAME  = 9.0    # layer names
FS_DIM   = 8.0    # dimensions / symbols / total
FS_SMALL = 7.5    # parameter counts, activation marks, footnotes, side labels
LW       = 0.8    # uniform stroke weight

# ---------------------------------------------------------------- geometry (inches)
Y_MID  = 2.40           # vertical centre line of all bars
H77    = 2.40           # height of the 77-wide input bar
BAR_W  = 0.16           # bar width
H_MIN  = 0.08           # minimum drawable bar height (the 1-wide output; true scale 0.03)
S      = 0.59           # centre-to-centre spacing of consecutive bars
S_OUT  = 0.66           # spacing fc6 -> out (a little more room for the "linear" mark)
MARGIN = 0.04           # white margin around the content

X_LBL   = 1.12                     # right edge of the input description labels
X_STRIP = X_LBL + 0.32             # z-score strip centre
STRIP_W = 0.26
X_IN    = X_STRIP + 0.46           # input bar centre
X_H     = [X_IN + S * (k + 1) for k in range(6)]
X_OUT   = X_H[-1] + S_OUT

def bar_h(n):
    return max(H77 * n / 77.0, H_MIN)

fig = plt.figure(figsize=(7.2, 4.0))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 7.2)
ax.set_ylim(0, 4.0)
ax.set_aspect("equal")
ax.axis("off")

bar_boxes = []   # (label, Bbox) for the overlap self-check
arrow_segs = []  # (x1, x2, y)

# ---------------------------------------------------------------- helpers
def rbar(xc, yb, w, h, color, z=3, rounding=None):
    r = rounding if rounding is not None else min(0.05, h / 2)
    p = FancyBboxPatch((xc - w / 2, yb), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=color, edgecolor="none", zorder=z)
    ax.add_patch(p)
    return p

def fan(x1r, y1b, y1t, x2l, y2b, y2t):
    ax.add_patch(Polygon([(x1r, y1b), (x2l, y2b), (x2l, y2t), (x1r, y1t)],
                         closed=True, facecolor=C_FAN, edgecolor="none", zorder=1))

def arrow(x1, x2, y, color=C_GREY, z=2):
    ax.annotate("", xy=(x2, y), xytext=(x1, y), zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=LW,
                                mutation_scale=7, shrinkA=0, shrinkB=0))
    arrow_segs.append((x1, x2, y))

def text(x, y, s, size=FS_SMALL, color=C_TEXT, ha="center", va="center", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, zorder=6, **kw)

# ---------------------------------------------------------------- layers
layers = [  # (name, width, x, colour, parameter count)
    ("input $x$", 77, X_IN,   C_RAIN, None),
    ("fc1",       56, X_H[0], C_HID,  "4,368"),
    ("fc2",       56, X_H[1], C_HID,  "3,192"),
    ("fc3",       56, X_H[2], C_HID,  "3,192"),
    ("fc4",       56, X_H[3], C_HID,  "3,192"),
    ("fc5",       56, X_H[4], C_HID,  "3,192"),
    ("fc6",       56, X_H[5], C_HID,  "3,192"),
    ("out",        1, X_OUT,  C_OUT,  "57"),
]
geom = []
for name, n, x, col, prm in layers:
    h = bar_h(n)
    geom.append((x, Y_MID - h / 2, Y_MID + h / 2))

# fans + centre-line arrows + activation marks between consecutive layers
for i in range(len(layers) - 1):
    x1, y1b, y1t = geom[i]
    x2, y2b, y2t = geom[i + 1]
    x1r, x2l = x1 + BAR_W / 2, x2 - BAR_W / 2
    fan(x1r, y1b, y1t, x2l, y2b, y2t)
    arrow(x1r + 0.02, x2l - 0.02, Y_MID)
    last = (i == len(layers) - 2)
    mark = "linear" if last else "ReLU"
    xm = (x1r + x2l) / 2 - (0.03 if last else 0.0)   # nudge "linear" into the wide end of its fan
    text(xm, Y_MID + 0.06, mark, va="bottom", color=C_TEXT)

# bars + labels
for (name, n, x, col, prm), (xx, yb, yt) in zip(layers, geom):
    h = yt - yb
    if n == 77:
        p = rbar(x, yb, BAR_W, h, C_RAIN)
        sh = H77 * 5 / 77.0
        r = Rectangle((x - BAR_W / 2, yb), BAR_W, sh, facecolor=C_STATIC,
                      edgecolor="none", zorder=3.1)
        ax.add_patch(r)
        r.set_clip_path(p)
    elif n == 1:
        rbar(x, yb, BAR_W, h, col, rounding=0.03)
    else:
        rbar(x, yb, BAR_W, h, col)
    bar_boxes.append((name, Bbox([[x - BAR_W / 2, yb], [x + BAR_W / 2, yt]])))
    # name above, dimension + parameter count below (tiny output bar gets more air)
    d_name, d_dim, d_prm = (0.07, 0.07, 0.22) if n > 1 else (0.20, 0.17, 0.32)
    text(x, yt + d_name, name, size=FS_NAME, va="bottom")
    text(x, yb - d_dim, r"$\mathbb{R}^{%d}$" % n, size=FS_DIM, va="top")
    if prm is not None:
        text(x, yb - d_prm, prm, va="top", color=C_GREY)

# ---------------------------------------------------------------- input side
x_in, yb_in, yt_in = geom[0]
y_split = yb_in + H77 * 5 / 77.0
y_rain_c = (y_split + yt_in) / 2
y_stat_c = (yb_in + y_split) / 2

# bracket on the left of the input bar (one piece per segment)
xb = x_in - BAR_W / 2 - 0.06
tick, gap = 0.045, 0.012
for (ya, yb_, col) in [(y_split + gap, yt_in, C_RAIN), (yb_in, y_split - gap, C_STATIC)]:
    ax.plot([xb, xb], [ya, yb_], color=col, lw=LW, solid_capstyle="butt", zorder=4)
    ax.plot([xb, xb + tick], [ya, ya], color=col, lw=LW, zorder=4)
    ax.plot([xb, xb + tick], [yb_, yb_], color=col, lw=LW, zorder=4)

# description labels + arrows that pass through the z-score strip
x_arrow_end = xb - 0.05
arrow(X_LBL + 0.05, x_arrow_end, y_rain_c, color=C_RAIN)
arrow(X_LBL + 0.05, x_arrow_end, y_stat_c, color=C_STATIC)
text(X_LBL, y_rain_c, "72 hourly rainfall\nhours $t{-}71,\\ldots,t$",
     ha="right", va="center", linespacing=1.3)
text(X_LBL, y_split,
     "5 catchment features\nat hour $t$:\nsm_2in, sm_20in,\nseason, T_air_C,\nsrad_Wm2",
     ha="right", va="top", linespacing=1.3)

# z-score strip (hollow, parameter-free op) drawn over the arrows
strip = FancyBboxPatch((X_STRIP - STRIP_W / 2, yb_in), STRIP_W, yt_in - yb_in,
                       boxstyle="round,pad=0,rounding_size=0.05",
                       facecolor="white", edgecolor=C_GREY, lw=LW, zorder=3)
ax.add_patch(strip)
t_strip = text(X_STRIP, Y_MID, r"z-score  $(x-\mu)/\sigma$", size=FS_DIM, rotation=90)
bar_boxes.append(("zscore", Bbox([[X_STRIP - STRIP_W / 2, yb_in],
                                  [X_STRIP + STRIP_W / 2, yt_in]])))

# ---------------------------------------------------------------- output side
x_o, yb_o, yt_o = geom[-1]
arrow(x_o + BAR_W / 2 + 0.03, x_o + BAR_W / 2 + 0.26, Y_MID)
text(x_o + BAR_W / 2 + 0.30, Y_MID,
     "$\\hat{Q}_t$ (mm/hr)\nstreamflow\nno activation",
     ha="left", va="center", linespacing=1.3)

# ---------------------------------------------------------------- h_k callout (fc3)
xk, ybk, ytk = geom[3]
cx, cy = xk + 0.42, ytk + 0.44
ax.plot([xk + BAR_W / 2, cx - 0.02], [ytk, cy - 0.02], color=C_GREY, lw=LW, zorder=5)
text(cx, cy, r"$h_k \in \mathbb{R}^{56}$", size=FS_NAME, ha="left", va="bottom")
text(cx, cy - 0.03, "post-ReLU activation, hidden layer $k$ = 1…6",
     ha="left", va="top", color=C_GREY)

# ---------------------------------------------------------------- footnote: tampered-layer context
xa, xb2 = X_H[1] - 0.10, X_H[5] + 0.10
yline = geom[1][1] - 0.22 - 0.16
ax.plot([xa, xa, xb2, xb2], [yline + 0.04, yline, yline, yline + 0.04],
        color=C_RULE, lw=LW, zorder=4)
t_foot = text((xa + xb2) / 2, yline - 0.06,
              "challenge: one of fc2–fc6 is tampered;\n"
              r"trigger direction $v \in \mathbb{R}^{56}$ in its input activations",
              va="top", color=C_GREY, linespacing=1.3)

# ---------------------------------------------------------------- self-check + auto-fit
def data_bbox(artist):
    return artist.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())

fig.canvas.draw()
# total parameter count: bottom right, one row below the footnote
fb = data_bbox(t_foot)
x_right = data_bbox(ax.texts[-1]).x1  # placeholder, replaced below
out_txt = [t for t in ax.texts if t.get_text().startswith("$\\hat{Q}")][0]
x_right = data_bbox(out_txt).x1
text(x_right, fb.y0 - 0.08, "total 20,385 parameters", size=FS_DIM,
     ha="right", va="top", color=C_TEXT)
fig.canvas.draw()

items = []
for t in ax.texts:
    if not t.get_text():
        continue  # empty annotation carriers
    items.append(("T:" + t.get_text().replace("\n", " ")[:30], data_bbox(t)))
items += [("B:" + n, bb) for n, bb in bar_boxes]
for (x1, x2, y) in arrow_segs:
    items.append(("A:%.2f-%.2f" % (x1, x2),
                  Bbox([[min(x1, x2), y - 0.005], [max(x1, x2), y + 0.005]])))
strip_names = {"B:zscore", "T:" + t_strip.get_text()[:30]}
problems = 0
for i in range(len(items)):
    for j in range(i + 1, len(items)):
        a, b = items[i], items[j]
        if not a[1].overlaps(b[1]):
            continue
        # arrows are deliberately hidden behind the white z-score strip
        if (a[0] in strip_names or b[0] in strip_names) and \
           (a[0].startswith("A:") or b[0].startswith("A:") or {a[0], b[0]} <= strip_names):
            continue
        problems += 1
        print("OVERLAP:", a[0], "<->", b[0])
print("overlap problems:", problems)

# content extent over every artist, then resize the canvas to content + margin
arts = [t for t in ax.texts if t.get_text()] + ax.patches + ax.lines
allbb = Bbox.union([data_bbox(a) for a in arts])
W = allbb.width + 2 * MARGIN
H = allbb.height + 2 * MARGIN
print("content extent (in): x %.2f..%.2f  y %.2f..%.2f  -> %.2f x %.2f (figure %.2f x %.2f)"
      % (allbb.x0, allbb.x1, allbb.y0, allbb.y1, allbb.width, allbb.height, W, H))
fig.set_size_inches(W, H)
ax.set_xlim(allbb.x0 - MARGIN, allbb.x1 + MARGIN)
ax.set_ylim(allbb.y0 - MARGIN, allbb.y1 + MARGIN)

fig.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0.01, facecolor="white")
print("saved", OUT)
