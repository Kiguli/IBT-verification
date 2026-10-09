"""Warehouse model -- multi-order fulfillment at industrial scale (the scalability run of Sec. IV-C).

A robot on a 21x21 floor with four rack columns must pick and place three orders within H = 280
ticks while a worker patrols an aisle and can block the robot's next cell; the robot's battery
drains by one per tick and is recharged at a charger, and a carried parcel can slip. The state is
(x, y, stage, battery, worker phase, charge flag), and seven interval leaves, Clopper-Pearson
intervals from the trial counts in COUNTS, drive the flat tick semantics of outcomes(). This
module implements BFS reachability, vectorized robust value iteration (interval /
O-maximization), leaf-midpoint DTMC value, Monte-Carlo bracket checks, and PRISM export of a
small variant. README.md lists which result files the paper's numbers come from.

Every edge stores its *leaf factorization* (product of leaf probabilities or
their complements), so static calibrations can be evaluated exactly and the
interval endpoints are products of leaf-interval endpoints.

Usage:
  python case3.py small           # (default) 11x11 validation variant + PRISM export
                                  # (writes results_case3_small.json)
  python case3.py full            # full-scale run (writes results_case3_full.json)
  python case3.py mc              # Monte-Carlo bracket checks (needs full run first)
  python case3.py materialize     # attempt full-Sigma tree-cached materialization
  python case3.py materialize-vi  # robust VI on the IMC materialized over all of Sigma
                                  # (writes results_case3_materialized_vi.json)
  python case3.py table           # print the leaf intervals computed from the trial counts
The modes wait, sweep, sweep2, tightened and horizon run further side experiments.
"""
from __future__ import annotations
import json, math, sys, time
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

# ---------------------------------------------------------------- calibration
# Per-leaf two-sided level alpha/L (alpha=0.05, L=7 leaves), so by Bonferroni
# the L leaf intervals hold SIMULTANEOUSLY with confidence >= 95%.
ALPHA = 0.05
NLEAVES = 7
ALPHA_PER_LEAF = ALPHA / NLEAVES
# z = Phi^{-1}(1 - alpha/(2L)) = 2.690110... (used by the Wilson interval)
Z_JOINT95 = 2.6901104976


def wilson(k, n, z=Z_JOINT95):
    """Wilson score interval. Approximate coverage; kept for comparison only."""
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def clopper_pearson(k, n, alpha=ALPHA_PER_LEAF):
    """Exact (Clopper-Pearson) binomial confidence interval at two-sided `alpha`.

    Guarantees coverage >= 1 - alpha for every true p, unlike the Wilson
    interval whose coverage oscillates below nominal. A certificate that is
    only as sound as its calibration needs the conservative choice, so this is
    what the paper reports.
    """
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.isf(alpha / 2, k + 1, n - k))
    return (lo, hi)


# The calibration used throughout: exact intervals at the Bonferroni-corrected
# level, so the whole leaf vector is covered with joint confidence >= 95%.
calib = clopper_pearson

COUNTS = {              # leaf -> (successes, trials)
    'pick_ok':        (183, 200),
    'place_ok':       (372, 390),
    'step_ok':        (1921, 2000),
    'step_blk':       (84, 200),
    'charge_tick_ok': (178, 190),
    'worker_advance': (312, 400),
    'slip':           (12, 600),
}
LEAVES = list(COUNTS)                       # leaf ids 0..6
LID = {name: i for i, name in enumerate(LEAVES)}
IVL = {name: calib(*COUNTS[name]) for name in COUNTS}
MID = {name: sum(IVL[name]) / 2 for name in COUNTS}


# ---------------------------------------------------------------- geometry
class Config:
    def __init__(self, small=False, policy='push'):
        self.policy = policy
        if small:                      # 11x11 validation variant: 2 racks, 1 order, no worker
            self.N = 11
            self.racks = [(3, y) for y in range(2, 9)] + [(7, y) for y in range(2, 9)]
            self.HOME = (0, 5); self.CHARGER = (0, 0)
            self.picks = [(4, 3)]; self.packs = [(10, 5)]
            self.patrol = []           # no worker
            self.BATMAX = 40; self.H = 90
            self.chg_lo, self.chg_hi, self.chg_gain = 12, 36, 9
        else:
            self.N = 21
            self.racks = [(x, y) for x in (3, 7, 13, 17) for y in range(3, 18)]
            self.HOME = (0, 10); self.CHARGER = (0, 0)
            self.picks = [(4, 5), (8, 12), (14, 7)]
            self.packs = [(20, 4), (20, 10), (20, 16)]
            self.patrol = [(10, 6), (10, 8), (10, 10), (10, 12), (10, 14),
                           (10, 12), (10, 10), (10, 8)]
            # Energy budget: chg_lo must exceed the max BFS distance to the
            # charger (~40 on this floor) or the robot strands mid-mission.
            self.BATMAX = 160; self.H = 280
            self.chg_lo, self.chg_hi, self.chg_gain = 45, 150, 9
        self.rackset = set(self.racks)
        self.NSTAGE = 2 * len(self.picks) + 1        # last stage = done
        self.NW = max(1, len(self.patrol))
        self.stations = [c for pair in zip(self.picks, self.packs) for c in pair]
        self.cells = [(x, y) for x in range(self.N) for y in range(self.N)
                      if (x, y) not in self.rackset]
        self.dist = {t: self._distfield(t) for t in set(self.stations + [self.CHARGER])}

    def _distfield(self, target):
        d = {target: 0}; q = deque([target])
        while q:
            (x, y) = q.popleft()
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < self.N and 0 <= ny < self.N and (nx, ny) not in self.rackset \
                        and (nx, ny) not in d:
                    d[(nx, ny)] = d[(x, y)] + 1
                    q.append((nx, ny))
        return d

    def next_cell(self, pos, target):
        d = self.dist[target]
        best = pos; bd = d[pos]
        for cand in ((pos[0] - 1, pos[1]), (pos[0] + 1, pos[1]),
                     (pos[0], pos[1] - 1), (pos[0], pos[1] + 1)):   # -x,+x,-y,+y tie-break
            if cand in d and d[cand] < bd:
                best, bd = cand, d[cand]
        return best


# ------------------------------------------------------- flat tick semantics
# An outcome: (successor_state, factors) with factors = tuple of (leaf_id, sign)
# sign +1 -> multiply by leaf value; sign -1 -> multiply by (1 - leaf value).
def chg_update(cfg, bat, stage, chg):
    if stage >= cfg.NSTAGE - 1:
        return 0
    if bat <= cfg.chg_lo:
        return 1
    if bat >= cfg.chg_hi:
        return 0
    return chg

def outcomes(cfg, s):
    (x, y, stage, bat, w, chg) = s
    done = cfg.NSTAGE - 1
    if stage == done or bat == 0:
        return [((x, y, stage, bat, w, chg), ())]          # absorbing
    pos = (x, y)
    robot = []                                             # (pos', stage', bat', factors)
    b = bat - 1
    station = cfg.stations[stage]
    carrying = (stage % 2 == 1)

    def emit(px, py, st, bb, fac):
        robot.append(((px, py), st, bb, fac))

    charging_tick = (chg == 1 and pos == cfg.CHARGER)
    slip_active = carrying and pos != station and not charging_tick
    slip_f = ((LID['slip'], +1),)
    noslip = ((LID['slip'], -1),) if slip_active else ()

    if slip_active:
        emit(x, y, stage - 1, b, slip_f)                   # parcel lost

    if charging_tick:
        emit(x, y, stage, min(cfg.BATMAX, bat + cfg.chg_gain),
             noslip + ((LID['charge_tick_ok'], +1),))
        emit(x, y, stage, b, noslip + ((LID['charge_tick_ok'], -1),))
    elif chg == 0 and pos == station:
        leaf = 'pick_ok' if stage % 2 == 0 else 'place_ok'
        emit(x, y, stage + 1, b, noslip + ((LID[leaf], +1),))
        emit(x, y, stage, b, noslip + ((LID[leaf], -1),))
    else:
        target = cfg.CHARGER if chg == 1 else station
        nxt = cfg.next_cell(pos, target)
        blocked = cfg.patrol and nxt == cfg.patrol[w]
        if blocked and cfg.policy == 'wait':
            emit(x, y, stage, b, noslip)          # deterministic wait
        else:
            leaf = 'step_blk' if blocked else 'step_ok'
            emit(nxt[0], nxt[1], stage, b, noslip + ((LID[leaf], +1),))
            emit(x, y, stage, b, noslip + ((LID[leaf], -1),))

    out = []
    wnext = (w + 1) % cfg.NW
    for (p2, st2, b2, fac) in robot:
        c2 = chg_update(cfg, b2, st2, chg)
        if cfg.patrol:
            out.append(((p2[0], p2[1], st2, b2, wnext, c2),
                        fac + ((LID['worker_advance'], +1),)))
            out.append(((p2[0], p2[1], st2, b2, w, c2),
                        fac + ((LID['worker_advance'], -1),)))
        else:
            out.append(((p2[0], p2[1], st2, b2, 0, c2), fac))
    return out


def edge_interval(factors):
    lo = hi = 1.0
    for (lid, sign) in factors:
        a, b = IVL[LEAVES[lid]]
        lo *= a if sign > 0 else 1 - b
        hi *= b if sign > 0 else 1 - a
    return lo, hi

def edge_value(factors, leafvals):
    p = 1.0
    for (lid, sign) in factors:
        v = leafvals[lid]
        p *= v if sign > 0 else 1 - v
    return p


# ------------------------------------------------------------------ build
def build(cfg):
    s0 = (cfg.HOME[0], cfg.HOME[1], 0, cfg.BATMAX, 0, 0)
    idx = {s0: 0}; order = [s0]; q = deque([s0])
    edges = []                                             # per state: list of (succ, factors)
    t0 = time.perf_counter()
    while q:
        s = q.popleft()
        outs = outcomes(cfg, s)
        for (t, fac) in outs:
            if t not in idx:
                idx[t] = len(order); order.append(t); q.append(t)
        edges.append([(idx[t], fac) for (t, fac) in outs])
    t_bfs = time.perf_counter() - t0
    n = len(order)
    K = max(len(e) for e in edges)
    succ = np.zeros((n, K), dtype=np.int32)
    lo = np.zeros((n, K)); hi = np.zeros((n, K))
    for i, es in enumerate(edges):
        for j, (t, fac) in enumerate(es):
            l, h = edge_interval(fac)
            succ[i, j] = t; lo[i, j] = l; hi[i, j] = h
    goal = np.array([s[2] == cfg.NSTAGE - 1 for s in order])
    return order, idx, edges, succ, lo, hi, goal, t_bfs


# -------------------------------------------------- vectorized robust VI
def robust_vi(succ, lo, hi, goal, H, lower=True, s0=0, record=None):
    n, K = succ.shape
    V = goal.astype(float)
    cap = hi - lo
    slack = 1.0 - lo.sum(axis=1)
    ng = ~goal
    for _ in range(H):
        sv = V[succ]                                       # (n, K)
        order = np.argsort(sv, axis=1, kind='stable')
        if not lower:
            order = order[:, ::-1]
        rows = np.arange(n)[:, None]
        cap_o = cap[rows, order]
        csum = np.cumsum(cap_o, axis=1) - cap_o
        take = np.clip(slack[:, None] - csum, 0.0, cap_o)
        q_o = lo[rows, order] + take
        Vn = (q_o * sv[rows, order]).sum(axis=1)
        V = np.where(ng, Vn, 1.0)
        if record is not None:
            record.append(float(V[s0]))
    return V

def point_vi(succ, P, goal, H, s0=0, record=None):
    V = goal.astype(float); ng = ~goal
    for _ in range(H):
        V = np.where(ng, (P * V[succ]).sum(axis=1), 1.0)
        if record is not None:
            record.append(float(V[s0]))
    return V

def factor_arrays(edges, n, K, maxf=4):
    fid = -np.ones((n, K, maxf), dtype=np.int8)
    fsg = np.zeros((n, K, maxf), dtype=np.int8)
    mask = np.zeros((n, K), dtype=bool)
    for i, es in enumerate(edges):
        mask[i, :len(es)] = True
        for j, (t, fac) in enumerate(es):
            assert len(fac) <= maxf
            for m, (lid, sign) in enumerate(fac):
                fid[i, j, m] = lid; fsg[i, j, m] = sign
    return fid, fsg, mask

def probs_for_vec(fid, fsg, mask, leafvals):
    lv = np.asarray(leafvals)
    vals = np.where(fid < 0, 1.0,
                    np.where(fsg > 0, lv[np.maximum(fid, 0)],
                             1.0 - lv[np.maximum(fid, 0)]))
    P = np.where(mask, vals.prod(axis=2), 0.0)
    err = np.abs(P.sum(axis=1) - 1.0).max()
    assert err < 1e-12, err
    return P

def probs_for(edges, n, K, leafvals):
    return probs_for_vec(*factor_arrays(edges, n, K), leafvals)


# ------------------------------------------------------------- Monte-Carlo
def rollout(succ, P, goal, H, s0, nroll, seed):
    rng = np.random.default_rng(seed)
    cum = np.cumsum(P, axis=1)
    state = np.full(nroll, s0, dtype=np.int64)
    done = goal[state].copy()
    for _ in range(H):
        u = rng.random(nroll)
        row = cum[state]
        nxt = (u[:, None] > row).sum(axis=1)
        state = succ[state, np.minimum(nxt, succ.shape[1] - 1)]
        done |= goal[state]
        if done.all():
            break
    return done.mean()


# ------------------------------------------------------------- PRISM export
def export_prism(order, edges, goal, H, stem):
    lines = [f"// case3 variant, |R|={len(order)}", "dtmc", "", "module M",
             f"  s : [0..{len(order)-1}] init 0;"]
    mids = [MID[l] for l in LEAVES]
    for i, es in enumerate(edges):
        terms = []
        for (t, fac) in es:
            l, h = edge_interval(fac)
            terms.append(f"[{l!r},{h!r}] : (s'={t})" if (l, h) != (1.0, 1.0)
                         else f"1 : (s'={t})")
        lines.append(f"  [] s={i} -> " + " + ".join(terms) + ";")
    lines += ["endmodule", "",
              'label "goal" = ' + " | ".join(f"s={i}" for i, g in enumerate(goal) if g) + ";", ""]
    (HERE / f"{stem}_imc.prism").write_text("\n".join(lines))
    lines = [f"// case3 variant midpoint DTMC", "dtmc", "", "module M",
             f"  s : [0..{len(order)-1}] init 0;"]
    for i, es in enumerate(edges):
        terms = [f"{edge_value(fac, mids)!r} : (s'={t})" for (t, fac) in es]
        lines.append(f"  [] s={i} -> " + " + ".join(terms) + ";")
    lines += ["endmodule", "",
              'label "goal" = ' + " | ".join(f"s={i}" for i, g in enumerate(goal) if g) + ";", ""]
    (HERE / f"{stem}_mid.prism").write_text("\n".join(lines))
    (HERE / f"{stem}.props.pctl").write_text(
        f'Pmin=? [ F<={H} "goal" ]\nPmax=? [ F<={H} "goal" ]\nP=? [ F<={H} "goal" ]\n')


# ------------------------------------------------------------------ drivers
def apply_count_mult(m, leaves=None):
    global IVL, MID
    for name in (leaves or COUNTS):
        k, n = COUNTS[name]
        IVL[name] = calib(m * k, m * n)
    MID.update({name: sum(IVL[name]) / 2 for name in COUNTS})

def run(small, do_export=False, policy='push', stem=None):
    cfg = Config(small=small, policy=policy)
    order, idx, edges, succ, lo, hi, goal, t_bfs = build(cfg)
    n, K = succ.shape
    sigma = (len(cfg.cells) * cfg.NSTAGE * (cfg.BATMAX + 1) * cfg.NW * 2)
    t0 = time.perf_counter(); Vlo = robust_vi(succ, lo, hi, goal, cfg.H, True)
    t_lo = time.perf_counter() - t0
    t0 = time.perf_counter(); Vhi = robust_vi(succ, lo, hi, goal, cfg.H, False)
    t_hi = time.perf_counter() - t0
    mids = [MID[l] for l in LEAVES]
    P = probs_for(edges, n, K, mids)
    t0 = time.perf_counter(); Vm = point_vi(succ, P, goal, cfg.H)
    t_mid = time.perf_counter() - t0
    res = {
        'variant': stem or ('small' if small else 'full'),
        'policy': policy,
        'sigma': sigma, 'R': n, 'K': K, 'H': cfg.H,
        'intervals': {l: IVL[l] for l in LEAVES},
        'lower': float(Vlo[0]), 'upper': float(Vhi[0]), 'midpoint': float(Vm[0]),
        't_bfs_s': round(t_bfs, 3), 't_vi_lower_s': round(t_lo, 3),
        't_vi_upper_s': round(t_hi, 3), 't_vi_midpoint_s': round(t_mid, 3),
        # The tree-direct side of the memory comparison. The paper's 5.5 GB baseline is measured
        # by run_materialize_vi through this same helper: same helper, same host, same
        # process-wide peak, so the two are like for like. The peak is taken AFTER BFS and every
        # VI sweep, so it covers the whole tree-direct pipeline.
        'peak_rss_gb': round(_peak_rss_mb() / 1000, 2),
        'peak_rss_method': _rss_method(),
    }
    print(json.dumps(res, indent=2))
    stem = stem or ('case3_small' if small else 'case3_full')
    (HERE / f"results_{stem}.json").write_text(json.dumps(res, indent=2))
    if do_export:
        export_prism(order, edges, goal, cfg.H, stem)
        print(f"PRISM files written: {stem}_imc.prism / {stem}_mid.prism")
    return cfg, order, idx, edges, succ, lo, hi, goal, res


def run_mc():
    cfg = Config(small=False)
    order, idx, edges, succ, lo, hi, goal, t_bfs = build(cfg)
    n, K = succ.shape
    fid, fsg, mask = factor_arrays(edges, n, K)
    mids = [MID[l] for l in LEAVES]
    res = json.loads((HERE / 'results_case3_full.json').read_text())
    P = probs_for_vec(fid, fsg, mask, mids)
    f_mid = rollout(succ, P, goal, cfg.H, 0, 200_000, seed=1)
    se = math.sqrt(f_mid * (1 - f_mid) / 200_000)
    print(f"MC midpoint: {f_mid:.5f} +- {2.58*se:.5f} (VI midpoint {res['midpoint']:.5f})",
          flush=True)
    rng = np.random.default_rng(7)
    viol = 0; freqs = []
    for k in range(20):
        leafvals = [rng.uniform(*IVL[l]) for l in LEAVES]
        Pk = probs_for_vec(fid, fsg, mask, leafvals)
        f = rollout(succ, Pk, goal, cfg.H, 0, 20_000, seed=100 + k)
        sek = math.sqrt(max(f * (1 - f), 1e-9) / 20_000)
        freqs.append(f)
        if not (res['lower'] - 3 * sek <= f <= res['upper'] + 3 * sek):
            viol += 1
        print(f"  pi_{k:02d}: freq={f:.5f}", flush=True)
    out = {'mc_midpoint': f_mid, 'mc_midpoint_ci99': 2.58 * se,
           'static_pi_freqs': freqs, 'bracket_violations': viol,
           'bracket': [res['lower'], res['upper']]}
    (HERE / 'results_case3_mc.json').write_text(json.dumps(out, indent=2))
    print(f"bracket violations: {viol}/20  (bracket [{res['lower']:.4f}, {res['upper']:.4f}])")


def run_tightened():
    """Verification-guided recalibration: quadruple the observation counts on the
    two widest leaves (step_blk, worker_advance) at the same success rate, and
    re-verify. Demonstrates how the certificate responds to targeted data
    collection."""
    global IVL
    for leaf in ('step_blk', 'worker_advance'):
        k, nn = COUNTS[leaf]
        IVL[leaf] = calib(4 * k, 4 * nn)
    cfg = Config(small=False)
    order, idx, edges, succ, lo, hi, goal, t_bfs = build(cfg)
    Vlo = robust_vi(succ, lo, hi, goal, cfg.H, True)
    Vhi = robust_vi(succ, lo, hi, goal, cfg.H, False)
    res = {'variant': 'tightened-4x-stepblk-workeradv',
           'intervals': {l: IVL[l] for l in LEAVES},
           'lower': float(Vlo[0]), 'upper': float(Vhi[0]), 'R': len(order)}
    print(json.dumps(res, indent=2))
    (HERE / 'results_case3_tightened.json').write_text(json.dumps(res, indent=2))


def _peak_rss_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except ImportError:
        import psutil
        return psutil.Process().memory_info().peak_wset / 1e6


def _rss_method():
    """Which branch of _peak_rss_mb produced the number, recorded beside it.

    The two branches do NOT share units, and only one of them is right. psutil's `peak_wset` is in
    BYTES, so /1e6 is MB -- correct. `ru_maxrss` is in KILOBYTES on Linux (bytes on macOS), so the
    same /1e6 yields GB on Linux and the result is 1000x too small once divided again for
    `peak_rss_gb`. Every figure in the paper was taken on the Windows benchmark host through the
    psutil branch and is correct; recording the method makes a re-run on Linux visibly suspect
    rather than silently wrong."""
    try:
        import resource  # noqa: F401
        return "resource.ru_maxrss (UNITS WRONG on Linux: KB/1e6 is GB, not MB)"
    except ImportError:
        return "psutil peak_wset (bytes/1e6 = MB)"


def run_materialize(budget_s=900):
    """Attempt tree-cached full-Sigma materialization (dict of per-state edge lists)."""
    cfg = Config(small=False)
    t0 = time.perf_counter(); count = 0; store = {}
    done = cfg.NSTAGE - 1
    try:
        for (x, y) in cfg.cells:
            for stage in range(cfg.NSTAGE):
                for bat in range(cfg.BATMAX + 1):
                    for w in range(cfg.NW):
                        for chg in (0, 1):
                            s = (x, y, stage, bat, w, chg)
                            store[s] = [(t, edge_interval(f)) for (t, f) in outcomes(cfg, s)]
                            count += 1
                            if count % 200_000 == 0:
                                el = time.perf_counter() - t0
                                mb = _peak_rss_mb()
                                print(f"  {count:,} states, {el:.0f}s, peak {mb:.0f} MB", flush=True)
                                if el > budget_s:
                                    raise TimeoutError
    except (TimeoutError, MemoryError) as e:
        el = time.perf_counter() - t0
        mb = _peak_rss_mb()
        print(f"STOPPED ({type(e).__name__}): {count:,}/{4267200:,} states "
              f"after {el:.0f}s, peak RSS {mb:.0f} MB")
        return
    el = time.perf_counter() - t0
    mb = _peak_rss_mb()
    print(f"COMPLETED: {count:,} states in {el:.0f}s, peak RSS {mb:.0f} MB")


def run_materialize_vi():
    """Robust VI on the fully materialized Sigma (the materialized-IMC baseline of Sec. IV-C)."""
    cfg = Config(small=False)
    t0 = time.perf_counter()
    all_states = [(x, y, st, b, w, c) for (x, y) in cfg.cells
                  for st in range(cfg.NSTAGE) for b in range(cfg.BATMAX + 1)
                  for w in range(cfg.NW) for c in (0, 1)]
    idx = {s: i for i, s in enumerate(all_states)}
    n = len(all_states)
    K = 6
    succ = np.zeros((n, K), dtype=np.int32)
    lo = np.zeros((n, K)); hi = np.zeros((n, K))
    for i, s in enumerate(all_states):
        for j, (t, fac) in enumerate(outcomes(cfg, s)):
            l, h = edge_interval(fac)
            succ[i, j] = idx[t]; lo[i, j] = l; hi[i, j] = h
    t_mat = time.perf_counter() - t0
    goal = np.array([s[2] == cfg.NSTAGE - 1 for s in all_states])
    s0 = idx[(cfg.HOME[0], cfg.HOME[1], 0, cfg.BATMAX, 0, 0)]
    t0 = time.perf_counter(); Vlo = robust_vi(succ, lo, hi, goal, cfg.H, True)
    t_lo = time.perf_counter() - t0
    t0 = time.perf_counter(); Vhi = robust_vi(succ, lo, hi, goal, cfg.H, False)
    t_hi = time.perf_counter() - t0
    res = {'variant': 'materialized-full-sigma', 'sigma': n,
           'lower': float(Vlo[s0]), 'upper': float(Vhi[s0]),
           't_materialize_s': round(t_mat, 1),
           't_vi_lower_s': round(t_lo, 1), 't_vi_upper_s': round(t_hi, 1),
           'peak_rss_gb': round(_peak_rss_mb() / 1000, 1),
           'calib': calib.__name__}
    print(json.dumps(res, indent=2))
    (HERE / 'results_case3_materialized_vi.json').write_text(json.dumps(res, indent=2))


def print_table():
    print(f"Clopper-Pearson at two-sided alpha/L = {ALPHA}/{NLEAVES} "
          f"= {ALPHA_PER_LEAF:.5f}; joint coverage >= {100*(1-ALPHA):.0f}% (Bonferroni)")
    for name, (k, n) in COUNTS.items():
        a, b = IVL[name]
        w = wilson(k, n)
        print(f"  {name:15s} {k:5d}/{n:<5d} -> [{a:.4f}, {b:.4f}]"
              f"   (Wilson [{w[0]:.4f}, {w[1]:.4f}])")


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'small'
    if mode == 'small':
        run(small=True, do_export=True)
    elif mode == 'full':
        run(small=False)
    elif mode == 'wait':
        run(small=False, policy='wait', stem='case3_wait')
    elif mode == 'sweep':
        for m in (4, 16):
            IVL.update({name: calib(*COUNTS[name]) for name in COUNTS})
            apply_count_mult(m)
            run(small=False, stem=f'case3_x{m}')
    elif mode == 'mc':
        run_mc()
    elif mode == 'tightened':
        run_tightened()
    elif mode == 'materialize':
        run_materialize()
    elif mode == 'materialize-vi':
        run_materialize_vi()
    elif mode == 'horizon':
        cfg = Config(small=False)
        order, idx, edges, succ, lo, hi, goal, t_bfs = build(cfg)
        rl, ru, rm = [], [], []
        robust_vi(succ, lo, hi, goal, cfg.H, True, record=rl)
        robust_vi(succ, lo, hi, goal, cfg.H, False, record=ru)
        P = probs_for(edges, *succ.shape, [MID[l] for l in LEAVES])
        point_vi(succ, P, goal, cfg.H, record=rm)
        out = {'H': cfg.H, 'lower': rl, 'upper': ru, 'midpoint': rm}
        (HERE / 'results_case3_horizon.json').write_text(json.dumps(out))
        print('horizon curves written; final:', rl[-1], ru[-1], rm[-1])
    elif mode == 'sweep2':
        for m in (2, 8):
            IVL.update({name: calib(*COUNTS[name]) for name in COUNTS})
            apply_count_mult(m)
            run(small=False, stem=f'case3_x{m}')
    elif mode == 'table':
        print_table()
