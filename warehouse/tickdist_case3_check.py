#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Does the compilation (tickdist) actually replace a hand-written model file?

``case3.py`` hand-writes ``outcomes(cfg, s)``: a per-state successor
function that enumerates, for one BT tick, every (successor, leaf
factorization) pair of the warehouse robot.  That function *is* the model file
that, with the compilation, a practitioner does not have to write.

This script declares the same behavior tree *declaratively* as an IBT, compiles
it with ``tickdist.tickdist``, and checks -- state by state, edge by edge,
factor by factor -- that the compiler reproduces the hand-written enumeration
exactly.  Nothing is compared numerically-with-tolerance: the monomials
(leaf_id, sign) tuples must be identical, in the same order.

    python tickdist_case3_check.py            # small config, all reachable states
    python tickdist_case3_check.py full       # + 4000 random and 4000 reachable states of the full config

Writes results_tickdist_case3.json.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'ibt'))

import case3                                                   # noqa: E402
from tickdist import (Action, Check, IntervalCheck, Selector, Sequence, SUC,
                      tickdist_full, leaf_boxes, check_wellformed)  # noqa: E402


# ------------------------------------------------------------------ the IBT --
def case3_ibt(cfg):
    """The tree of case3.py, written declaratively instead of by hand.

    State s = (x, y, stage, bat, w, chg), exactly as in case3.py.
    """
    done = cfg.NSTAGE - 1
    IVL = case3.IVL

    def pos(s):
        return (s[0], s[1])

    def station(s):
        return cfg.stations[s[2]]

    def carrying(s):
        return s[2] % 2 == 1

    def charging_tick(s):
        return s[5] == 1 and pos(s) == cfg.CHARGER

    def slip_active(s):
        return carrying(s) and pos(s) != station(s) and not charging_tick(s)

    def at_station(s):
        return s[5] == 0 and pos(s) == station(s)

    def target(s):
        return cfg.CHARGER if s[5] == 1 else station(s)

    def blocked(s):
        return bool(cfg.patrol) and cfg.next_cell(pos(s), target(s)) == \
            cfg.patrol[s[4]]

    def ipchk(name):
        return IntervalCheck(name, *IVL[name])

    # --- updates (all keep the tuple layout of case3.py) -------------------
    def drop(s):
        return (s[0], s[1], s[2] - 1, s[3] - 1, s[4], s[5])

    def charge(s):
        return (s[0], s[1], s[2], min(cfg.BATMAX, s[3] + cfg.chg_gain),
                s[4], s[5])

    def burn(s):
        return (s[0], s[1], s[2], s[3] - 1, s[4], s[5])

    def advance(s):
        return (s[0], s[1], s[2] + 1, s[3] - 1, s[4], s[5])

    def step(s):
        n = cfg.next_cell(pos(s), target(s))
        return (n[0], n[1], s[2], s[3] - 1, s[4], s[5])

    def wnext(s):
        return (s[0], s[1], s[2], s[3], (s[4] + 1) % cfg.NW, s[5])

    def chgupd(s):
        return (s[0], s[1], s[2], s[3], s[4],
                case3.chg_update(cfg, s[3], s[2], s[5]))

    def try_or(leaf, ok_update, else_update):
        """Selector(Sequence(ipchk, act_ok), act_else) -- always returns Suc."""
        return Selector(Sequence(ipchk(leaf), Action(leaf + '_ok', ok_update)),
                        Action(leaf + '_no', else_update, SUC))

    # --- the three mutually exclusive robot cases -------------------------
    charge_case = Sequence(Check('charging', charging_tick),
                           try_or('charge_tick_ok', charge, burn))
    station_case = Sequence(
        Check('at_station', at_station),
        Selector(Sequence(Check('even_stage', lambda s: s[2] % 2 == 0),
                          try_or('pick_ok', advance, burn)),
                 try_or('place_ok', advance, burn)))
    if cfg.policy == 'wait':
        blocked_case = Sequence(Check('blocked', blocked),
                                Action('wait', burn, SUC))
    else:
        blocked_case = Sequence(Check('blocked', blocked),
                                try_or('step_blk', step, burn))
    move_case = Selector(blocked_case, try_or('step_ok', step, burn))

    robot = Selector(
        Sequence(Check('slip_active', slip_active), ipchk('slip'),
                 Action('drop', drop)),
        Selector(charge_case, station_case, move_case))

    worker = (Selector(Sequence(ipchk('worker_advance'),
                                Action('advance_worker', wnext)),
                       Action('hold_worker', lambda s: s, SUC))
              if cfg.patrol else Action('no_worker', lambda s: s, SUC))

    return Selector(
        Sequence(Check('absorbing', lambda s: s[2] == done or s[3] == 0),
                 Action('idle', lambda s: s, SUC)),
        Sequence(robot, worker, Action('chg_update', chgupd)))


# ------------------------------------------------------------- comparison ---
def handwritten(cfg, s):
    """case3.outcomes with leaf ids mapped to names."""
    return [(t, tuple((case3.LEAVES[lid], sg) for (lid, sg) in fac))
            for (t, fac) in case3.outcomes(cfg, s)]


def compiled(tree, s):
    return tickdist_full(tree, s, check=True).imc_edges()


def compare(cfg, tree, states, stop_after=5):
    diffs = []
    n_edges = 0
    for s in states:
        a = handwritten(cfg, s)
        b = compiled(tree, s)
        n_edges += len(a)
        if a != b:
            if sorted(map(str, a)) == sorted(map(str, b)):
                diffs.append({'state': str(s), 'kind': 'order-only'})
            else:
                diffs.append({'state': str(s), 'kind': 'mismatch',
                              'handwritten': [str(x) for x in a],
                              'tickdist': [str(x) for x in b]})
            if len(diffs) >= stop_after:
                break
    return diffs, n_edges


def verify_from_compiler(cfg, tree):
    """Compile the IBT with tickdist.compile_imc, then run case3's OWN robust VI on it.

    The model handed to the verifier is produced by the compiler; nothing about
    the state space or the edge intervals is written by hand.  The published
    numbers of results_case3_small.json must come back unchanged.
    """
    import numpy as np
    from tickdist import compile_imc

    t0 = time.perf_counter()
    imc = compile_imc(tree, (cfg.HOME[0], cfg.HOME[1], 0, cfg.BATMAX, 0, 0))
    t_compile = time.perf_counter() - t0
    n = imc.n()
    K = max(len(r) for r in imc.succ)
    succ = np.zeros((n, K), dtype=np.int32)
    lo = np.zeros((n, K))
    hi = np.zeros((n, K))
    for i in range(n):
        for j, t in enumerate(imc.succ[i]):
            succ[i, j] = t
            lo[i, j] = float(imc.lo[i][j])
            hi[i, j] = float(imc.hi[i][j])
    goal = np.array([s[2] == cfg.NSTAGE - 1 for s in imc.states])
    Vlo = case3.robust_vi(succ, lo, hi, goal, cfg.H, lower=True)
    Vhi = case3.robust_vi(succ, lo, hi, goal, cfg.H, lower=False)
    # midpoint DTMC, using the compiler's factorization
    mid = [case3.MID[nm] for nm in case3.LEAVES]
    P = np.zeros((n, K))
    for i in range(n):
        col = {t: j for j, t in enumerate(imc.succ[i])}
        for (t, m) in imc.edges[i]:
            p = 1.0
            for (lid, sg) in m:
                v = mid[case3.LID[lid]]
                p *= v if sg > 0 else 1 - v
            P[i, col[t]] += p
        assert abs(P[i].sum() - 1.0) < 1e-12, P[i].sum()
    Vm = case3.point_vi(succ, P, goal, cfg.H)
    return {'n_states': n, 'K': K, 'H': cfg.H,
            'lower': float(Vlo[0]), 'upper': float(Vhi[0]),
            'midpoint': float(Vm[0]),
            'compile_seconds': t_compile,
            'tickdist_calls': imc.calls}


def reachable(cfg, limit=None):
    s0 = (cfg.HOME[0], cfg.HOME[1], 0, cfg.BATMAX, 0, 0)
    seen = {s0}
    q = deque([s0])
    order = [s0]
    while q:
        s = q.popleft()
        for (t, _f) in case3.outcomes(cfg, s):
            if t not in seen:
                seen.add(t)
                q.append(t)
                order.append(t)
                if limit and len(order) >= limit:
                    return order
    return order


def main(argv):
    full = 'full' in argv[1:]
    res = {'configs': []}
    for tag, cfg in (('small', case3.Config(small=True)),
                     ('small_wait', case3.Config(small=True, policy='wait'))) + \
            ((('full', case3.Config(small=False)),) if full else ()):
        tree = case3_ibt(cfg)
        wf = check_wellformed(tree, strict=False)
        t0 = time.perf_counter()
        if tag == 'full':
            rng = random.Random(7)
            free = [c for c in cfg.cells]
            states = []
            for _ in range(4000):
                x, y = rng.choice(free)
                states.append((x, y, rng.randrange(cfg.NSTAGE),
                               rng.randrange(0, cfg.BATMAX + 1),
                               rng.randrange(cfg.NW), rng.randrange(2)))
            states += reachable(cfg, limit=4000)
        else:
            states = reachable(cfg)
        diffs, n_edges = compare(cfg, tree, states)
        dt = time.perf_counter() - t0
        rec = {'config': tag, 'policy': cfg.policy, 'n_states': len(states),
               'n_edges': n_edges, 'identical': not diffs,
               'diffs': diffs[:5], 'seconds': dt,
               'leaf_ids_used': sorted(leaf_boxes(tree)),
               'leaf_wellformedness_ok': wf['ok']}
        res['configs'].append(rec)
        print('%-11s %6d states %7d edges  identical=%s  (%.2fs)'
              % (tag, len(states), n_edges, not diffs, dt))
        for d in diffs[:2]:
            print('   diff', json.dumps(d)[:400])
    # end-to-end: the published small-variant numbers, recomputed from the
    # compiler's model instead of the hand-written one.
    cfg = case3.Config(small=True)
    got = verify_from_compiler(cfg, case3_ibt(cfg))
    pub_path = os.path.join(HERE,
                            'results_case3_small.json')
    pub = json.load(open(pub_path)) if os.path.exists(pub_path) else {}
    devs = {k: abs(got[k] - pub[k]) for k in ('lower', 'upper', 'midpoint')
            if k in pub}
    res['end_to_end'] = {'from_tickdist': got,
                         'published_handwritten': {k: pub.get(k) for k in
                                                   ('R', 'K', 'H', 'lower',
                                                    'upper', 'midpoint')},
                         'abs_deviation': devs,
                         'bit_identical': all(d == 0.0 for d in devs.values()),
                         'source': pub_path}
    print('end-to-end   compiled model -> robust VI: '
          '[%.15f, %.15f] mid %.15f'
          % (got['lower'], got['upper'], got['midpoint']))
    print('             published hand-written    : '
          '[%.15f, %.15f] mid %.15f  identical=%s'
          % (pub.get('lower', float('nan')), pub.get('upper', float('nan')),
             pub.get('midpoint', float('nan')),
             res['end_to_end']['bit_identical']))

    res['ok'] = all(c['identical'] for c in res['configs']) and \
        res['end_to_end']['bit_identical']
    res['meta'] = {'case3': os.path.abspath(case3.__file__),
                   'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'python': sys.version.split()[0]}
    out = os.path.join(HERE, 'results_tickdist_case3.json')
    with open(out, 'w') as fh:
        json.dump(res, fh, indent=1, default=str)
    print('wrote %s' % out)
    return 0 if res['ok'] else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
