#!/usr/bin/env python
r"""Synthetic case study: one IBT, its certified interval, and N simulated runs that land inside it.

    python synth_case.py            # certify, simulate, write results_synth.json (about 1 min)

THE MISSION. A mobile manipulator delivers K=2 orders from a shelf (cell 0) to a handover station
(cell 12) within H=80 ticks, crossing a pedestrian zone (cells 5-7) on every trip. One person
detector serves two contexts: at the crossing a detection makes the robot YIELD (presence hurts);
at the station a detection is REQUIRED for the handover (presence helps). It is Fig. 1's
pickup-and-deliver grown by one shared leaf, and its grasp and handover intervals are Fig. 1's pick
[.70,.90] and deliver [.85,.95] verbatim.

WHAT MAKES THE AGREEMENT MEAN SOMETHING. The prediction and the outcome come from different code.

  prediction  robust VI on compile_imc(T) -- the compilation's Extend, interval products and capped
              Minkowski sums -- and, for a fixed nature, exact forward propagation of the tree's
              one-step law.
  outcome     `execute`, an interpreter written from Table I: each tick starts at the root, the
              composites are literal loops, an interval leaf draws Bernoulli(value) only when it is
              ticked. It imports the node CLASSES and nothing that compiles: no compilation, no
              interval arithmetic, no compiled chain.

A check that sampled the compiled chain itself would agree with point VI by
construction. This one can disagree: an unsound compilation or a wrong edge polynomial would put the
executed tree somewhere the certificate does not.

THE NATURES, fixed in advance with their seeds:
  * 36 fixed calibrations (one value per leaf for the whole run): all 16 box corners, and 20
    interior draws. These are leaf assignments, theta in Theta(T) (Sec. IV-B).
  * A, the adversary, written from the tree's MEANING before any VI output is consulted: the person
    is present at the crossing as often as the box allows and at the station as rarely, every other
    leaf at its low end. It is not a fixed calibration -- the one detector takes two values in two
    places -- and it is covered because the certificate admits any within-box drift of the leaf
    probabilities during execution (Sec. IV-B). It should land ON the certified floor.
  * W, a plausible world: presence 0.48 at the crossing and 0.22 at the station, every value
    interior to its box. It is what an honest calibration could produce and no fixed-calibration
    analysis can represent. For contrast, `pooled_sbt_H` is the point-probability SBT a pooled
    detector rate (0.35) and W's other values would give.
"""
import itertools
import json
import math
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ibt"))
import tickdist as td  # noqa: E402
from tickdist import (Action, Branch, Check, IntervalAction, IntervalCheck,  # noqa: E402
                      Selector, Sequence, FAIL, RUN, SUC)
from scipy.stats import chi2 as chi2_dist  # noqa: E402

# ------------------------------------------------------------------------------- the mission --
L, K, H = 12, 2, 80
XING = {5, 6, 7}
BOX = {"person": (0.20, 0.50), "handover": (0.85, 0.95), "grasp": (0.70, 0.90), "move": (0.80, 0.95)}
LEAVES = list(BOX)
S0 = (0, 0, 0)                                  # (cell, carrying, delivered)
N = 20_000                                      # episodes per nature
SEEDS = {"interior": 20260929, "episodes": 7}   # fixed before any run


def TGT(s):
    return s[2] == K


def goal(s):
    return L if s[1] else 0


def step(s):
    return (s[0] + (1 if goal(s) > s[0] else -1), s[1], s[2])


def build():
    """The IBT, using only the nodes of Table I (reactive tick semantics).

    The grasp retry is ?(grasp, grasp): a second attempt within the tick if the first fails. It is
    the same leaf twice, so one calibration value serves both attempts -- the parameter sharing of
    Sec. IV-B.
    """
    B = BOX
    grasp = lambda: IntervalAction("grasp", [Branch(SUC, lambda s: (s[0], 1, s[2]), *B["grasp"]),  # noqa: E731
                                             Branch(FAIL, lambda s: s)])
    return Selector(
        Check("done", lambda s: s[2] == K),
        Sequence(Check("at_station", lambda s: s[0] == L), Check("carrying", lambda s: s[1] == 1),
                 IntervalCheck("person", *B["person"]),
                 IntervalAction("handover", [Branch(SUC, lambda s: (s[0], s[1], s[2] + 1), *B["handover"]),
                                             Branch(FAIL, lambda s: s)]),
                 Action("release", lambda s: (s[0], 0, s[2]))),
        Sequence(Check("at_shelf", lambda s: s[0] == 0), Check("empty", lambda s: s[1] == 0),
                 Selector(grasp(), grasp())),
        Sequence(Check("in_crossing", lambda s: s[0] in XING),
                 IntervalCheck("person", *B["person"]),
                 Action("yield", lambda s: s, RUN)),
        Sequence(Check("away", lambda s: s[0] != goal(s)),
                 IntervalAction("move", [Branch(SUC, step, *B["move"]), Branch(FAIL, lambda s: s)])))


# ------------------------------------------------------ the independent executor (outcomes) --
def execute(node, s, value, rng):
    """Tick `node` once at state `s`. `value(leaf, s)` is the nature's success probability."""
    if isinstance(node, Check):
        return (SUC if node.pred(s) else FAIL), s
    if isinstance(node, Action):
        return node.status, node.update(s)
    if isinstance(node, IntervalCheck):
        return (SUC if rng.random() < value(node.name, s) else FAIL), s
    if isinstance(node, IntervalAction):
        b_suc, b_fail = node.branches
        b = b_suc if rng.random() < value(node.name, s) else b_fail
        return b.status, b.update(s)
    if isinstance(node, Sequence):
        for c in node.children():
            st, s = execute(c, s, value, rng)
            if st != SUC:
                return st, s
        return SUC, s
    if isinstance(node, Selector):
        for c in node.children():
            st, s = execute(c, s, value, rng)
            if st != FAIL:
                return st, s
        return FAIL, s
    raise TypeError(type(node))


def simulate(tree, value, n, seed):
    """Empirical Pr[delivered by t], t = 0..H, from n executed episodes."""
    rng = random.Random(seed)
    first = [0] * (H + 2)
    for _ in range(n):
        s, hit = S0, H + 1
        for t in range(H):
            _, s = execute(tree, s, value, rng)
            if TGT(s):
                hit = t + 1
                break
        first[hit] += 1
    cdf, c = [0.0], 0
    for k in range(1, H + 1):
        c += first[k]
        cdf.append(c / n)
    return cdf


# ------------------------------------------------------------- the verifier side (predictions) --
def predict(tree, imc, value):
    """Exact Pr[delivered by t] under a state-dependent nature, by forward propagation of the
    tree's one-step law (the compilation in point mode) -- the verifier's prediction for that nature."""
    i0 = imc.index[S0]
    V = [1.0 if TGT(s) else 0.0 for s in imc.states]
    out = [V[i0]]
    for _ in range(1, H + 1):
        Vn = []
        for s in imc.states:
            if TGT(s):
                Vn.append(1.0)
                continue
            d = td.tickdist(tree, s, {l: value(l, s) for l in LEAVES}, check=False)
            Vn.append(sum(p * V[imc.index[s2]] for (_st, s2), (p, _) in d.items()))
        V = Vn
        out.append(V[i0])
    return out


def main():
    T = build()
    t0 = time.perf_counter()
    imc = td.compile_imc(T, S0)
    i0 = imc.index[S0]
    _, tlo = td.robust_vi(imc, TGT, H, True)
    _, thi = td.robust_vi(imc, TGT, H, False)
    t_verify = time.perf_counter() - t0
    Vlo = [r[i0] for r in tlo]
    Vhi = [r[i0] for r in thi]

    # --- the natures, fixed in advance
    natures = []
    for corner in itertools.product(*[BOX[l] for l in LEAVES]):
        cal = dict(zip(LEAVES, corner))
        natures.append(("fixed", "corner", cal, (lambda c: lambda l, s: c[l])(cal)))
    rng = random.Random(SEEDS["interior"])
    for _ in range(20):
        cal = {l: rng.uniform(*BOX[l]) for l in LEAVES}
        natures.append(("fixed", "interior", cal, (lambda c: lambda l, s: c[l])(cal)))
    P = BOX["person"]
    A = lambda l, s: (P[1] if s[0] in XING else P[0]) if l == "person" else BOX[l][0]  # noqa: E731
    Wv = {"handover": 0.87, "grasp": 0.75, "move": 0.83}
    W = lambda l, s: (0.48 if s[0] in XING else 0.22) if l == "person" else Wv[l]  # noqa: E731
    natures.append(("adversary", "A", None, A))
    natures.append(("world", "W", None, W))

    # --- simulate every nature from the executor, and predict it from the verifier
    dkw = math.sqrt(math.log(2 / 0.05) / (2 * N))   # whole-CDF 95% band per nature
    t1 = time.perf_counter()
    rows = []
    for j, (kind, name, cal, value) in enumerate(natures):
        emp = simulate(T, value, N, SEEDS["episodes"] + j)
        pred = predict(T, imc, value)
        ph = pred[H]
        se = math.sqrt(max(ph * (1 - ph), 1e-12) / N)
        inside = all(Vlo[t] - dkw <= emp[t] <= Vhi[t] + dkw for t in range(H + 1))
        rows.append({"kind": kind, "name": name, "cal": cal, "emp": emp, "pred": pred,
                     "emp_H": emp[H], "pred_H": ph, "z_H": (emp[H] - ph) / se,
                     "max_cdf_gap": max(abs(e - p) for e, p in zip(emp, pred)),
                     "inside_band": inside})
    t_sim = time.perf_counter() - t1

    fixed = [r for r in rows if r["kind"] == "fixed"]
    adv = next(r for r in rows if r["name"] == "A")
    wld = next(r for r in rows if r["name"] == "W")
    chi2 = sum(r["z_H"] ** 2 for r in fixed)
    mid = {l: sum(BOX[l]) / 2 for l in LEAVES}
    res = {
        "mission": {"L": L, "K": K, "H": H, "crossing": sorted(XING), "box": BOX},
        "model": {"sigma": (L + 1) * 2 * (K + 1), "R": imc.n(), "verify_s": round(t_verify, 4),
                  "max_out_degree": max(len(x) for x in imc.succ)},
        "certificate_H": [Vlo[H], Vhi[H]],
        "Vlo": Vlo, "Vhi": Vhi,
        "midpoint_H": td.point_vi(imc, mid, TGT, H)[0][i0],
        "N": N, "seeds": SEEDS, "episodes_total": N * len(natures), "simulate_s": round(t_sim, 1),
        "dkw_95": dkw,
        "fixed": {"n": len(fixed),
                  "range_H": [min(r["pred_H"] for r in fixed), max(r["pred_H"] for r in fixed)],
                  "all_inside_band": all(r["inside_band"] for r in fixed),
                  "max_abs_z_H": max(abs(r["z_H"]) for r in fixed),
                  "chi2_H": chi2, "dof": len(fixed), "chi2_p": float(chi2_dist.sf(chi2, len(fixed))),
                  "max_cdf_gap": max(r["max_cdf_gap"] for r in fixed)},
        "adversary": {"emp_H": adv["emp_H"], "pred_H": adv["pred_H"], "z_H": adv["z_H"],
                      "pred_equals_floor_maxdiff": max(abs(a - b) for a, b in zip(adv["pred"], Vlo)),
                      "inside_band": adv["inside_band"]},
        "world": {"emp_H": wld["emp_H"], "pred_H": wld["pred_H"], "z_H": wld["z_H"],
                  "inside_band": wld["inside_band"],
                  "pooled_sbt_H": td.point_vi(imc, {"person": 0.35, **Wv}, TGT, H)[0][i0]},
        "rows": rows,
    }
    (HERE / "results_synth.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "Vlo", "Vhi")}, indent=1))


if __name__ == "__main__":
    main()
