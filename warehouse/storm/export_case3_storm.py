"""Export the warehouse model (case3.py) for Storm / storm-pars, in two different ways.

(A) PRUNED, state-by-state.  Reuses ``case3.build()`` verbatim (the same BFS
    reachable set the paper's robust VI solves, 671,532 states) and writes it as
    a flat Storm DRN file -- interval-valued (``@value_type: double-interval``)
    for Storm's native interval-MC engine, and rational-function-valued
    (``@value_type: parametric``) for storm-pars parameter lifting.
    This measures "what does Storm cost *once somebody has already built the
    model for it*".

(B) FULL, symbolic.  A PRISM module over the actual state variables
    ``x, y, stage, bat, w, chg`` with guarded commands, so that *Storm itself*
    performs reachability and model construction over the 6,870,192-state
    product.  This is the analogue of what the paper's pipeline does from the
    behavior tree: build the reachable model from a compact description.

Why DRN and not a PRISM state-enumeration for (A): a PRISM module with one
guarded command per state costs Storm |S| guard evaluations per state, i.e.
O(|S|^2) = 4.5e11 for 671,532 states.  That would measure our encoding, not
Storm.  DRN is Storm's own linear-time explicit input format, so (A) is the most
favourable possible reading for Storm.

Geometry note for (B).  ``next_cell`` is a greedy descent on a BFS distance
field, which is data, not a closed-form expression -- exactly as it is data in
the Python model.  It is encoded as guarded commands whose guards are
rectangle-compressed cell regions (one command family per
(stage, chg, direction)), which keeps the command count in the low hundreds so
Storm's per-state guard evaluation is cheap.  Everything that constitutes the
*state space* -- the product with stage/battery/worker/charge-flag and the
reachability -- is left entirely to Storm.

Outputs land in ``build_storm/`` next to this script.

Usage::

    python export_case3_storm.py --variant full     # the real thing
    python export_case3_storm.py --variant medium   # 11x11 + patrol, validation
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASE3_DIR = HERE.parent
sys.path.insert(0, str(CASE3_DIR))

import case3  # noqa: E402

# leaf id -> storm-pars parameter name
PNAME = {
    "pick_ok": "p_pick",
    "place_ok": "p_place",
    "step_ok": "p_step",
    "step_blk": "p_stepblk",
    "charge_tick_ok": "p_charge",
    "worker_advance": "p_wadv",
    "slip": "p_slip",
}
PARAMS = [PNAME[l] for l in case3.LEAVES]


# --------------------------------------------------------------- edge labels
def poly(factors):
    """Parametric probability string for a leaf factorization."""
    if not factors:
        return "1"
    parts = []
    for (lid, sign) in factors:
        p = PNAME[case3.LEAVES[lid]]
        parts.append(p if sign > 0 else "(1-%s)" % p)
    return "*".join(parts)


def ivl(factors):
    """Interval probability string [lo,hi] for a leaf factorization."""
    if not factors:
        return "1"
    lo, hi = case3.edge_interval(factors)
    return "[%r,%r]" % (lo, hi)


def region_string():
    return ",".join("%r<=%s<=%r" % (case3.IVL[l][0], PNAME[l], case3.IVL[l][1])
                    for l in case3.LEAVES)


def midpoint_string():
    return ",".join("%s=%r" % (PNAME[l], case3.MID[l]) for l in case3.LEAVES)


# ============================================================== (A) DRN, pruned
def write_drn(path, order, edges, goal, parametric, s0=0):
    """Write the pruned chain as a Storm DRN file.

    Interval variant uses ``@value_type: double-interval`` and ``dst : [lo,hi]``.
    Parametric variant uses ``@value_type: parametric`` with a ``@placeholders``
    table, since the warehouse model has only a few dozen distinct leaf factorizations.
    """
    n = len(order)
    # distinct factorizations -> string (and, parametric only, placeholder id)
    seen = {}
    for es in edges:
        for (_t, fac) in es:
            if fac not in seen:
                seen[fac] = None
    keys = sorted(seen, key=lambda f: (len(f), f))
    if parametric:
        ph = {f: "$%d" % i for i, f in enumerate(keys)}
        label = {f: ph[f] for f in keys}
    else:
        label = {f: ivl(f) for f in keys}

    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write("// Case 3 pruned reachable chain, exported from case3.build()\n")
        fh.write("@type: DTMC\n")
        fh.write("@value_type: %s\n" % ("parametric" if parametric else "double-interval"))
        fh.write("@parameters\n")
        fh.write((" ".join(PARAMS) + " \n") if parametric else "\n")
        if parametric:
            fh.write("@placeholders\n")
            for f in keys:
                fh.write("%s : (%s)/(1)\n" % (ph[f], poly(f)))
        fh.write("@reward_models\n\n")
        fh.write("@nr_states\n%d\n" % n)
        fh.write("@nr_choices\n%d\n" % n)
        fh.write("@model\n")
        buf = []
        for i, es in enumerate(edges):
            tags = ""
            if i == s0:
                tags += " init"
            if goal[i]:
                tags += " goal"
            buf.append("state %d%s\n\taction 0\n" % (i, tags))
            for (t, fac) in es:
                buf.append("\t\t%d : %s\n" % (t, label[fac]))
            if len(buf) > 200000:
                fh.write("".join(buf))
                buf = []
        fh.write("".join(buf))
    return len(keys)


# ========================================================= (B) symbolic PRISM
def rect_cover(cells):
    """Greedy maximal-rectangle cover of a set of grid cells."""
    S = set(cells)
    rects = []
    while S:
        x0, y0 = min(S, key=lambda c: (c[1], c[0]))
        x1 = x0
        while (x1 + 1, y0) in S:
            x1 += 1
        y1 = y0
        while all((xx, y1 + 1) in S for xx in range(x0, x1 + 1)):
            y1 += 1
        rects.append((x0, x1, y0, y1))
        for xx in range(x0, x1 + 1):
            for yy in range(y0, y1 + 1):
                S.discard((xx, yy))
    return rects


def region_expr(cells):
    parts = []
    for (x0, x1, y0, y1) in rect_cover(cells):
        a = "x=%d" % x0 if x0 == x1 else "x>=%d&x<=%d" % (x0, x1)
        b = "y=%d" % y0 if y0 == y1 else "y>=%d&y<=%d" % (y0, y1)
        parts.append("(%s&%s)" % (a, b))
    return "(" + "|".join(parts) + ")" if len(parts) > 1 else parts[0]


def wset_expr(ws, nw):
    """Guard for `w in ws` / its complement, whichever is shorter."""
    ws = sorted(set(ws))
    if len(ws) * 2 > nw:
        comp = [i for i in range(nw) if i not in ws]
        inner = "|".join("w=%d" % i for i in comp)
        return "!(%s)" % inner if comp else "true", \
               ("(%s)" % inner if comp else "false")
    inner = "|".join("w=%d" % i for i in ws)
    return "(%s)" % inner, "!(%s)" % inner


def _upd(parts):
    return "&".join(parts)


def symbolic_prism(cfg, parametric, naive=False):
    """PRISM module over (x,y,stage,bat,w,chg); Storm does the reachability.

    ``naive=True`` emits one guarded command per (stage, chg, cell) instead of
    per rectangle-compressed region -- ~30x more commands.  It exists purely to
    show that the headline build time is not an artefact of the guard
    compression: Storm's per-state cost is linear in the command count, so the
    naive encoding is the pessimistic reading of the same experiment.
    """
    L = case3.LID
    DONE = cfg.NSTAGE - 1
    patrol_idx = {}
    for i, p in enumerate(cfg.patrol):
        patrol_idx.setdefault(p, []).append(i)

    out = [
        "// Case 3 -- FULL symbolic model.  Storm performs reachability and",
        "// model construction over the %d-state product itself."
        % (len(cfg.cells) * cfg.NSTAGE * (cfg.BATMAX + 1) * cfg.NW * 2),
        "dtmc",
        "",
        "const int BMAX = %d;" % cfg.BATMAX,
        "const int CLO  = %d;   // charge if battery <= CLO" % cfg.chg_lo,
        "const int CHI  = %d;   // stop charging if battery >= CHI" % cfg.chg_hi,
        "const int GAIN = %d;" % cfg.chg_gain,
        "const int DONE = %d;" % DONE,
        "",
    ]
    if parametric:
        for l in case3.LEAVES:
            lo, hi = case3.IVL[l]
            out.append("const double %s;   // region [%r, %r]" % (PNAME[l], lo, hi))
        out.append("")
    out += [
        "formula bm  = bat-1;",
        "formula bc  = min(BMAX, bat+GAIN);",
        "// chg' after a normal tick (stage stays, and stage < DONE)",
        "formula cs  = (bm<=CLO ? 1 : (bm>=CHI ? 0 : chg));",
        "// chg' after a successful charging tick",
        "formula cch = (bc<=CLO ? 1 : (bc>=CHI ? 0 : chg));",
        "// chg' after the stage advances",
        "formula cnx = (stage+1>=DONE ? 0 : (bm<=CLO ? 1 : (bm>=CHI ? 0 : chg)));",
        "",
        "module M",
        "  x     : [0..%d] init %d;" % (cfg.N - 1, cfg.HOME[0]),
        "  y     : [0..%d] init %d;" % (cfg.N - 1, cfg.HOME[1]),
        "  stage : [0..%d] init 0;" % DONE,
        "  bat   : [0..%d] init %d;" % (cfg.BATMAX, cfg.BATMAX),
        "  w     : [0..%d] init 0;" % (cfg.NW - 1),
        "  chg   : [0..1] init 0;",
        "",
    ]

    pr = poly if parametric else ivl
    ncmd = 0

    def worker(robot):
        """(factors, update) list -> worker-advance product."""
        res = []
        for fac, upd in robot:
            if cfg.patrol:
                res.append((fac + ((L["worker_advance"], +1),),
                            _upd(upd + ["(w'=mod(w+1,%d))" % cfg.NW])))
                res.append((fac + ((L["worker_advance"], -1),), _upd(upd)))
            else:
                res.append((fac, _upd(upd)))
        return res

    def emit(guard, robot):
        nonlocal ncmd
        terms = " + ".join("%s : %s" % (pr(f), u) for f, u in worker(robot))
        out.append("  [] %s -> %s;" % (guard, terms))
        ncmd += 1

    # ---- absorbing -------------------------------------------------------
    out.append("  // absorbing: mission complete, or battery empty")
    out.append("  [] (stage=DONE)|(bat=0) -> 1 : (bat'=bat);")
    ncmd += 1
    out.append("")

    # ---- charging tick ---------------------------------------------------
    cx, cy = cfg.CHARGER
    out.append("  // charging tick at the charger")
    emit("stage<DONE & bat>0 & chg=1 & x=%d & y=%d" % (cx, cy),
         [(((L["charge_tick_ok"], +1),), ["(bat'=bc)", "(chg'=cch)"]),
          (((L["charge_tick_ok"], -1),), ["(bat'=bm)", "(chg'=cs)"])])
    out.append("")

    # ---- manipulation at the current station -----------------------------
    out.append("  // pick / place at the station of the current stage")
    for st in range(DONE):
        sx, sy = cfg.stations[st]
        leaf = "pick_ok" if st % 2 == 0 else "place_ok"
        emit("stage=%d & bat>0 & chg=0 & x=%d & y=%d" % (st, sx, sy),
             [(((L[leaf], +1),), ["(stage'=stage+1)", "(bat'=bm)", "(chg'=cnx)"]),
              (((L[leaf], -1),), ["(bat'=bm)", "(chg'=cs)"])])
    out.append("")

    # ---- movement --------------------------------------------------------
    out.append("  // movement: greedy descent on the BFS distance field to the")
    out.append("  // current target (station if chg=0, charger if chg=1)")
    for st in range(DONE):
        for cv in (0, 1):
            target = cfg.stations[st] if cv == 0 else cfg.CHARGER
            groups = {}
            for c in cfg.cells:
                if c == target:
                    continue
                nxt = cfg.next_cell(c, target)
                d = (nxt[0] - c[0], nxt[1] - c[1])
                assert d != (0, 0), (c, target)
                slip = (st % 2 == 1) and (c != cfg.stations[st])
                ws = tuple(patrol_idx.get(nxt, ()))
                groups.setdefault((d, slip, ws), []).append(c)
            if naive:                       # one command per cell, no compression
                groups = {(k[0], k[1], k[2], c): [c]
                          for k, cs in groups.items() for c in cs}
            for key, cells in sorted(groups.items()):
                (d, slip, ws) = key[0], key[1], key[2]
                dx, dy = d
                base = "stage=%d & bat>0 & chg=%d & %s" % (st, cv, region_expr(cells))
                mv = []
                if dx:
                    mv.append("(x'=x%+d)" % dx)
                if dy:
                    mv.append("(y'=y%+d)" % dy)

                def branches(blocked):
                    stp = L["step_blk"] if blocked else L["step_ok"]
                    robot = []
                    if slip:
                        robot.append((((L["slip"], +1),),
                                      ["(stage'=stage-1)", "(bat'=bm)", "(chg'=cs)"]))
                        ns = ((L["slip"], -1),)
                    else:
                        ns = ()
                    robot.append((ns + ((stp, +1),), mv + ["(bat'=bm)", "(chg'=cs)"]))
                    robot.append((ns + ((stp, -1),), ["(bat'=bm)", "(chg'=cs)"]))
                    return robot

                if ws:
                    inw, notw = wset_expr(ws, cfg.NW)
                    emit(base + " & " + inw, branches(True))
                    emit(base + " & " + notw, branches(False))
                else:
                    emit(base, branches(False))

    out += ["endmodule", "", 'label "goal" = stage=DONE;', ""]
    return "\n".join(out), ncmd


# ------------------------------------------------------------------- driver
def make_cfg(variant):
    if variant == "full":
        return case3.Config(small=False)
    cfg = case3.Config(small=True)          # 11x11, but give it a patrol so the
    cfg.patrol = [(1, 5), (2, 1), (5, 1), (5, 9), (2, 9)]   # worker/blocking
    cfg.NW = len(cfg.patrol)                # branches are exercised too
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="full", choices=("full", "medium"))
    ap.add_argument("--out-dir", type=Path, default=HERE / "build_storm")
    ap.add_argument("--skip-drn", action="store_true")
    ap.add_argument("--skip-vi", action="store_true")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = "case3" if args.variant == "full" else "case3med"

    cfg = make_cfg(args.variant)
    timings = {}

    # ---- (B) symbolic PRISM ------------------------------------------------
    for parametric, tag in ((False, "imc"), (True, "pmc")):
        t0 = time.perf_counter()
        txt, ncmd = symbolic_prism(cfg, parametric)
        (args.out_dir / ("%s_sym_%s.prism" % (stem, tag))).write_text(txt)
        timings["t_write_sym_%s_s" % tag] = round(time.perf_counter() - t0, 3)
        timings["sym_%s_commands" % tag] = ncmd
        timings["sym_%s_bytes" % tag] = len(txt)
    t0 = time.perf_counter()
    txt, ncmd = symbolic_prism(cfg, False, naive=True)
    (args.out_dir / ("%s_symnaive_imc.prism" % stem)).write_text(txt)
    timings["t_write_symnaive_imc_s"] = round(time.perf_counter() - t0, 3)
    timings["symnaive_imc_commands"] = ncmd
    (args.out_dir / ("%s.props.pctl" % stem)).write_text(
        'P=? [ F<=%d "goal" ]\n' % cfg.H)
    (args.out_dir / ("%s.region" % stem)).write_text(region_string() + "\n")

    # ---- (A) pruned DRN ----------------------------------------------------
    t0 = time.perf_counter()
    order, idx, edges, succ, lo, hi, goal, t_bfs = case3.build(cfg)
    timings["t_python_build_s"] = round(time.perf_counter() - t0, 3)
    timings["t_python_bfs_s"] = round(t_bfs, 3)
    n = len(order)
    nedge = sum(len(e) for e in edges)
    timings["pruned_states"] = n
    timings["pruned_edges"] = nedge
    timings["sigma"] = int(len(cfg.cells) * cfg.NSTAGE * (cfg.BATMAX + 1) * cfg.NW * 2)
    timings["horizon"] = int(cfg.H)
    timings["num_goal_states"] = int(goal.sum())

    if not args.skip_vi:
        t0 = time.perf_counter()
        Vlo = case3.robust_vi(succ, lo, hi, goal, cfg.H, True)
        timings["t_ours_vi_lower_s"] = round(time.perf_counter() - t0, 3)
        t0 = time.perf_counter()
        Vhi = case3.robust_vi(succ, lo, hi, goal, cfg.H, False)
        timings["t_ours_vi_upper_s"] = round(time.perf_counter() - t0, 3)
        timings["reference_lower"] = float(Vlo[0])
        timings["reference_upper"] = float(Vhi[0])

    if not args.skip_drn:
        for parametric, tag in ((False, "imc"), (True, "pmc")):
            p = args.out_dir / ("%s_pruned_%s.drn" % (stem, tag))
            t0 = time.perf_counter()
            nk = write_drn(p, order, edges, goal, parametric)
            timings["t_write_drn_%s_s" % tag] = round(time.perf_counter() - t0, 3)
            timings["drn_%s_bytes" % tag] = p.stat().st_size
            timings["distinct_factorizations"] = nk

    timings["variant"] = args.variant
    timings["region"] = region_string()
    timings["midpoint_constants"] = midpoint_string()
    timings["property"] = 'P=? [ F<=%d "goal" ]' % cfg.H
    (args.out_dir / ("%s_export.json" % stem)).write_text(json.dumps(timings, indent=2))
    print(json.dumps(timings, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
