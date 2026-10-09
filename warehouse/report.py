"""Print the numbers of the scalability run of Sec. IV-C from the result files in this folder.
Reads only; writes nothing.

    python report.py              # timings: median of the last 3 runs of each mode
    python report.py --last 5

Timings and memory of the compiled pipeline are medians over the last N runs of
`python timed_pipeline.py td_dropin` in timed_runs.jsonl (the paper uses N = 3). The materialized
baseline is the single run in results_case3_materialized_vi.json. A file that does not exist yet is
skipped.
"""
import argparse
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))


def load(rel):
    path = os.path.join(HERE, rel)
    if not os.path.exists(path):
        print('  (%s not found, skipped)' % rel)
        return None
    with open(path) as fh:
        return json.load(fh)


def runs_by_mode(last):
    path = os.path.join(HERE, 'timed_runs.jsonl')
    if not os.path.exists(path):
        print('  (timed_runs.jsonl not found, skipped)')
        return {}
    by = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                by.setdefault(rec['mode'], []).append(rec)
    return {m: rs[-last:] for m, rs in by.items()}


def med(rs, key):
    return statistics.median(r[key] for r in rs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--last', type=int, default=3, help='runs per mode to take the median of')
    args = ap.parse_args()

    print('== model (results_case3_full.json, model_facts.json)')
    full = load('results_case3_full.json')
    if full:
        print('  |Sigma| = %d   |R| = %d   H = %d   max out-degree K = %d'
              % (full['sigma'], full['R'], full['H'], full['K']))
        print('  certified interval [%.3f, %.3f]  (%r, %r)'
              % (full['lower'], full['upper'], full['lower'], full['upper']))
        for leaf, (a, b) in full['intervals'].items():
            print('    %-15s [%.4f, %.4f]' % (leaf, a, b))
    facts = load('model_facts.json')
    if facts:
        f = facts['sigma_factorisation']
        print('  factorisation %d free cells x %d stages x %d battery x %d worker x %d charge = %d'
              % (f['passable_cells'], f['stages'], f['battery_levels'], f['worker_phases'],
                 f['charge_flag_values'], facts['sigma']))
        print('  Clopper-Pearson at alpha/L = %g/%d; all intervals equal results_case3_full.json: %s'
              % (facts['alpha'], facts['n_leaves'], facts['all_intervals_match']))

    print('\n== compiled pipeline (timed_runs.jsonl, median of the last %d runs per mode)' % args.last)
    runs = runs_by_mode(args.last)
    for mode in ('td_dropin', 'hand', 'td_compile'):
        rs = runs.get(mode)
        if not rs:
            continue
        ident = all(r['lower_bit_identical'] and r['upper_bit_identical'] for r in rs)
        print('  %-10s n=%d  BFS %.3f s   VI %.3f / %.3f s   BFS+2VI %.3f s   peak after 2 VI %.2f GB'
              '   bit-identical %s'
              % (mode, len(rs), med(rs, 't_bfs_s'), med(rs, 't_vi_lower_s'),
                 med(rs, 't_vi_upper_s'), med(rs, 't_bfs_plus_two_vi_s'),
                 med(rs, 'peak_rss_gb_after_two_vi'), ident))
    print('  (the paper reports td_dropin: the compilation is the BFS edge oracle)')

    print('\n== materialized baseline (results_case3_materialized_vi.json)')
    mat = load('results_case3_materialized_vi.json')
    if mat:
        total = mat['t_materialize_s'] + mat['t_vi_lower_s'] + mat['t_vi_upper_s']
        print('  |Sigma| = %d   build %.1f s   VI %.1f / %.1f s   total %.1f s   peak %.1f GB'
              % (mat['sigma'], mat['t_materialize_s'], mat['t_vi_lower_s'], mat['t_vi_upper_s'],
                 total, mat['peak_rss_gb']))
        if full:
            print('  same interval as results_case3_full.json, bit for bit: %s'
                  % (mat['lower'] == full['lower'] and mat['upper'] == full['upper']))
        rs = runs.get('td_dropin')
        if rs:
            ours = med(rs, 't_bfs_plus_two_vi_s')
            mem = med(rs, 'peak_rss_gb_after_two_vi')
            print('  time %.1f s vs %.1f s = %.2fx   memory %.1f GB vs %.2f GB = %.2fx'
                  % (total, ours, total / ours, mat['peak_rss_gb'], mem, mat['peak_rss_gb'] / mem))

    print('\n== compilation vs hand-compiled successor function (equiv_full.json)')
    eq = load('equiv_full.json')
    if eq:
        print('  states checked %d of %d reachable, identical %d, mismatches %d'
              % (eq['n_states_checked'], eq['n_reachable_states'], eq['n_states_identical'],
                 eq['n_states_mismatch']))
        print('  edges %d (hand) = %d (compiled); leaf factors compared %d'
              % (eq['n_edges_handwritten'], eq['n_edges_tickdist'], eq['n_leaf_factors_compared']))
        b, c = eq['tickdist_bfs'], eq['tickdist_compile_imc']
        print('  BFS with the compilation as oracle: same states in same order %s, arrays bit-identical %s'
              % (b['same_states_same_order'], b['arrays_succ_lo_hi_bit_identical']))
        print('  compile_imc intervals bit-identical to the hand-compiled ones: %s'
              % c['per_successor_intervals_bit_identical_to_case3'])

    print('\n== Storm (storm/storm_at_scale.json)')
    st = load(os.path.join('storm', 'storm_at_scale.json'))
    if st:
        print('  interval DRN file %.1f MB (%d bytes)'
              % (st['export']['drn_imc_bytes'] / 1e6, st['export']['drn_imc_bytes']))
        r1 = st.get('run1_pruned_interval_mc', {})
        for d in ('min', 'max'):
            r = r1.get(d, {})
            print('  %s: value %s   Storm wallclock %s s   peak %s MB   states %s'
                  % (d, r.get('value'), r.get('storm_wallclock_s'), r.get('peak_memory_mb'),
                     r.get('states')))
        v = st.get('validation', {})
        print('  abs error vs certified interval: %s / %s'
              % (v.get('run1_min_abs_error'), v.get('run1_max_abs_error')))

    print('\n== corners of Theta(T) (results_case3_corners.json)')
    co = load('results_case3_corners.json')
    if co:
        print('  %d corners attain [%.3f, %.3f]; certified interval conservative by at most %.3f / %.3f'
              % (co['n_corners'], co['corner_min'], co['corner_max'], co['slack_lower'],
                 co['slack_upper']))


if __name__ == '__main__':
    main()
