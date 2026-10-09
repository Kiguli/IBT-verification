#!/usr/bin/env python
"""The case-study figure, from results_synth.json:  python fig_synth.py  ->  fig_synth.pdf

One axis: Pr[both orders delivered within t ticks] against the deadline t.
  grey band      the certified interval [V_lo(t), V_hi(t)] -- one robust-VI run gives every t <= H
  thin grey      the 36 fixed leaf assignments, simulated (20,000 episodes each)
  blue circles   adversary A, simulated; its predicted curve IS the band's lower edge
  orange squares world W, simulated; dashed orange, its exact prediction
Identity survives grayscale: circle vs square, solid band edge vs dashed line. Labels sit at the
right end of each curve in the text colour, not the series colour.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
d = json.loads((HERE / "results_synth.json").read_text())
H = d["mission"]["H"]
t = list(range(H + 1))
fixed = [r for r in d["rows"] if r["kind"] == "fixed"]
A = next(r for r in d["rows"] if r["name"] == "A")
W = next(r for r in d["rows"] if r["name"] == "W")

INK, MUTED = "#1f2328", "#5b6068"
BAND, EDGE, FIXED = "#e4e6ea", "#9aa0a8", "#7d838c"
BLUE, ORANGE = "#2a78d6", "#eb6834"       # two categorical colours, checked for colour-vision deficiency

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
})
fig, ax = plt.subplots(figsize=(3.4, 2.15))
fig.subplots_adjust(left=0.135, right=0.765, bottom=0.185, top=0.975)

T0 = 40
ts = t[T0:]
ax.fill_between(ts, d["Vlo"][T0:], d["Vhi"][T0:], color=BAND, lw=0, zorder=1)
for v in (d["Vlo"], d["Vhi"]):
    ax.plot(ts, v[T0:], color=EDGE, lw=0.8, zorder=2)
for r in fixed:
    ax.plot(ts, r["emp"][T0:], color=FIXED, lw=0.45, alpha=0.55, zorder=3)
ax.plot(ts, W["pred"][T0:], color=ORANGE, lw=0.9, ls=(0, (3, 1.6)), zorder=4)
mk = list(range(50, H + 1, 2))
ax.plot(mk, [W["emp"][k] for k in mk], ls="none", marker="s", ms=3.2, mfc=ORANGE, mec="white",
        mew=0.5, zorder=5)
ax.plot(mk, [A["emp"][k] for k in mk], ls="none", marker="o", ms=3.4, mfc=BLUE, mec="white",
        mew=0.5, zorder=5)

ax.set_xlim(T0, H + 0.6)
ax.set_ylim(0, 1.02)
ax.set_xticks(range(T0, H + 1, 10))
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax.set_yticklabels(["0", ".25", ".50", ".75", "1"])
# Axis labels are typeset in the paper's LaTeX (Fig. 3(c)), in the paper's notation:
# x "deadline $H$ (ticks)" centred at (110.2, 9.0) pt, y "$\mathbb{P}_{T_\rho}[\phi](s_0)$" at (10.9, 89.8) pt.
# (y label: see above)
ax.grid(axis="y", color="#eceef1", lw=0.5, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(length=2, width=0.6, pad=1.5)


def end_label(y, text, dy=0.0):
    ax.annotate(text, xy=(H + 0.6, y), xytext=(H + 2.2, y + dy), textcoords="data",
                annotation_clip=False, va="center", ha="left", fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.4, shrinkA=0, shrinkB=1))


fmin, fmax = d["fixed"]["range_H"]
# Curve ends can sit close together (at H=80 all three lie above .85), so stack the labels top-down
# with room for their line counts; each leader still points at its curve's end.
LINE = 0.072                                     # one 8.5 pt text line in data units on this axis
labels = sorted([(0.5 * (fmin + fmax), "36 leaf\nassignments", 2), (W["emp_H"], "world $W$", 1),
                 (A["emp_H"], "adversary $A$", 1)], key=lambda x: -x[0])
prev_bottom = None
for y, text, lines in labels:
    half = lines * LINE / 2
    ty = y if prev_bottom is None else min(y, prev_bottom - half - 0.035)
    end_label(y, text, ty - y)
    prev_bottom = ty - half
ax.annotate("certified\ninterval", xy=(50.5, 0.62), xytext=(44.0, 0.85), fontsize=8.5, color=INK,
            ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.4, shrinkA=1, shrinkB=0))

fig.savefig(HERE / "fig_synth.pdf")
print("wrote fig_synth.pdf")
