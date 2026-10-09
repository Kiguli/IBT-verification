"""Print the numbers of Sec. V from the result files in this folder. Reads only; writes nothing.

    python report.py

Most numbers are fields of results_synth.json. Three are derived here, because no script stores
them: the largest excursion of a simulated curve outside the certified band, the length of the
chain (the first deadline with a nonzero upper bound), and the gap between the certificate's floor
and the minimum over the leaf assignments. synth_grid.json and synth_person_sweep.json back the
claims about the minimum over Theta(T); a file that does not exist yet is skipped.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(rel):
    path = os.path.join(HERE, rel)
    if not os.path.exists(path):
        print('  (%s not found, skipped)' % rel)
        return None
    with open(path) as fh:
        return json.load(fh)


def main():
    d = load('results_synth.json')
    if d is None:
        return
    m, H = d['mission'], d['mission']['H']
    rows = d['rows']
    Vlo, Vhi = d['Vlo'], d['Vhi']

    print('== mission and certificate')
    print('  station cell %d, orders K = %d, H = %d, pedestrian zone cells %s'
          % (m['L'], m['K'], H, m['crossing']))
    print('  |Sigma| = %d   |R| = %d   max out-degree %d'
          % (d['model']['sigma'], d['model']['R'], d['model']['max_out_degree']))
    print('  certified interval at H = %d: [%.3f, %.6f]  (%r, %r)'
          % (H, d['certificate_H'][0], d['certificate_H'][1], *d['certificate_H']))
    print('  compile + both robust VI sweeps: %.4f s (one run)' % d['model']['verify_s'])
    print('  iterates stored for every deadline k = 0..%d: %d lower, %d upper'
          % (H, len(Vlo), len(Vhi)))
    first = next(t for t in range(H + 1) if Vhi[t] > 0)
    print('  first deadline with a nonzero upper bound (chain length): %d' % first)

    print('\n== simulation')
    kinds = {}
    for r in rows:
        key = r['name'] if r['kind'] != 'fixed' else r['kind'] + ' ' + r['name']
        kinds[key] = kinds.get(key, 0) + 1
    print('  %d episodes per strategy of nature, seeds %s' % (d['N'], d['seeds']))
    print('  %d strategies: %s' % (len(rows), ', '.join('%s %d' % kv for kv in kinds.items())))
    print('  95%% DKW margin %.4f' % d['dkw_95'])
    exc = []
    for r in rows:
        for t in range(H + 1):
            exc.append((max(Vlo[t] - r['emp'][t], r['emp'][t] - Vhi[t]), r['name'], t))
    exc.sort(reverse=True)
    e, who, t = exc[0]
    print('  largest excursion outside the certified band: %.4f by %s at t = %d' % (e, who, t))
    nxt = next(x for x in exc if x[1] != who)
    print('  largest by any other strategy: %.2g by %s at t = %d' % nxt)
    print('  every curve inside band +- DKW margin: %s' % all(r['inside_band'] for r in rows))

    f, a, w = d['fixed'], d['adversary'], d['world']
    print('\n== leaf assignments (%d)' % f['n'])
    print('  exact values at H: %.3f - %.3f' % tuple(f['range_H']))
    print('  largest gap simulated vs exact, over all H: %.4f' % f['max_cdf_gap'])
    print('  chi-square of the deviations at H: %.1f, dof %d, p-value %.2f'
          % (f['chi2_H'], f['dof'], f['chi2_p']))

    print('\n== adversary A and world W')
    print('  A: exact equals the lower iterate for every H to within %.1e; simulated %.4f (exact %.4f)'
          % (a['pred_equals_floor_maxdiff'], a['emp_H'], a['pred_H']))
    print('  W: simulated %.3f, exact %.3f; pooled-detector SBT (0.35) predicts %.3f'
          % (w['emp_H'], w['pred_H'], w['pooled_sbt_H']))
    print('  gap between certificate floor and min over leaf assignments: %.4f - %.4f = %.2f'
          % (f['range_H'][0], d['certificate_H'][0], f['range_H'][0] - d['certificate_H'][0]))

    print('\n== minimum over Theta(T) (synth_grid.json, synth_person_sweep.json)')
    g = load('synth_grid.json')
    if g:
        print('  %d-point grid: min %.4f at %s (corner: %s, attained by %d point(s))'
              % (g['n_assignments'], g['min'], g['argmin'], g['argmin_is_corner'],
                 g['n_assignments_attaining_min']))
        for leaf, mono in g['per_leaf_monotonicity_on_grid'].items():
            print('    %-9s nondecreasing on the grid: %s' % (leaf, mono['nondecreasing_everywhere']))
        print('  gap floor to grid minimum: %.2f' % g['gap_floor_to_grid_min'])
    s = load('synth_person_sweep.json')
    if s:
        lo = s['sweeps']['others_low']
        print('  detector sweep, other leaves at lower ends: min %.4f at detector %.2f'
              % (lo['min'], lo['argmin_person']))


if __name__ == '__main__':
    main()
