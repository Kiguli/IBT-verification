# IBT-verification

This repository accompanies the paper

> Ben Wooding and Taylor T. Johnson, **Interval Behavior Trees and Their Verification**.

It holds the code and result files for the paper's case study, a simulated delivery mission
(Sec. V), and for the warehouse model behind the scalability run in Sec. IV-C. Every number in
Sec. V and every warehouse number in the paper comes from a script here. The result file that each
script writes is committed, so you can compare a rerun against it.

## Citation

TBD.

## License

The code is released under the MIT License (`LICENSE`): you may use, modify and redistribute it as
you please, provided the copyright notice is kept.

## The case study and the scalability run

**Warehouse (the scalability run in Sec. IV-C, `warehouse/`).** A robot must fulfill three
pick-and-place orders within H = 280 ticks on a 21x21 floor while a worker patrols. The blackboard
has |Σ| = 6,870,192 states, and seven interval leaves carry Clopper–Pearson intervals computed
from illustrative trial counts (`COUNTS` in `case3.py`), not from real trials. The run
shows that Algorithm 1 (BFS with the compilation as the edge oracle, then robust value iteration on
the reachable states only) returns the same certified interval as robust VI on the IMC materialized
over all of Σ, in a fraction of the time and memory. Storm checks that interval independently, and
the 128 corners of the leaf box show how conservative it is.

**Delivery (Sec. V, `delivery/`).** A mobile manipulator delivers two orders from a shelf to a
handover station across a pedestrian zone, and one person detector serves two places in the tree
(|Σ| = 78). The study certifies the delivery probability for every deadline H ≤ 80. It then tests
the certificate independently of the compilation: an interpreter written from the node semantics
runs the tree for 20,000 episodes under each of 38 strategies of nature.

## Layout

```
ibt/
  tickdist.py              the IBT library: node classes, the compilation to an IMC, robust VI
warehouse/                 the scalability run in Sec. IV-C
  case3.py                 the warehouse model: hand-compiled successor function, BFS,
                           robust VI, the materialized baseline
  tickdist_case3_check.py  the same robot written as an IBT for the compilation
  equiv_full.py            compilation vs hand-compiled successor function on all of R
  timed_pipeline.py        timed BFS + robust VI with the compilation as the edge oracle
  corners.py               the values at the 128 corners of the leaf box (point VI)
  model_facts.py           factorisation of |Σ|, leaf intervals from the trial counts
  hw.py                    description of the host (Windows only)
  exp_common.py            path setup for equiv_full.py, timed_pipeline.py, model_facts.py
                           and hw.py
  report.py                prints the warehouse numbers from the result files
  storm/                   the Storm check (Docker)
  *.json, *.jsonl, *.log   the result files the warehouse numbers come from
delivery/                  Sec. V
  synth_case.py            the delivery IBT, its certificate, the interpreter, the simulations
  fig_synth.py             Fig. 3(c)
  synth_grid.py            9^4 grid over the leaf box: minimum and monotonicity
  synth_person_sweep.py    dense sweep of the person detector
  exp_common.py            path setup for the two scripts above
  report.py                prints the Sec. V numbers from the result files
  *.json                   the result files the paper's numbers come from
```

`tickdist.py` uses only the Python standard library. `case3.py` needs NumPy and SciPy,
`fig_synth.py` needs matplotlib, and peak memory is read with psutil.

## Setup

Use Python 3.12 (the paper used 3.12.10). If `python` on your PATH is another version, call the
3.12 interpreter by name or path in the first line.

```bash
python -m venv .venv
source .venv/bin/activate          # Git Bash on Windows: source .venv/Scripts/activate
                                   # cmd on Windows:      .venv\Scripts\activate
pip install -r requirements.txt
```

The Storm check also needs Docker. `warehouse/storm/run_storm_at_scale.py` runs the image
`movesrwth/storm:stable`. That tag moves, so pull the image the paper used (Storm 1.13.0) by its
digest and give it the tag the script expects:

```bash
docker pull movesrwth/storm@sha256:82bba7377c16f6e362853b0c77aa89dde7a3f07c3db38d38a6c8e251105d3669
docker tag  movesrwth/storm@sha256:82bba7377c16f6e362853b0c77aa89dde7a3f07c3db38d38a6c8e251105d3669 movesrwth/storm:stable
```

## Reproducing the numbers

Each script writes its results next to itself, so `git diff` shows what a rerun changed. All
values are deterministic except timings and memory. Runtimes in the tables are for the paper's
host: Intel Core Ultra 9 285K, 64 GB RAM, Windows 11, Python 3.12.10. The long steps run on one
core (pure Python BFS, NumPy VI). Timings depend on what else the machine is doing. In reruns on
the same host while other jobs were running, the Python steps took up to 30% longer than the
committed timings and the Storm calls up to 50% longer. Every value and every peak-memory figure
was unchanged.

After any rerun, `python report.py` in each folder prints every number of its study from the
result files. That includes the numbers derived from them: medians, ratios, the largest excursion,
the chain length and the 0.11 gap.

Apart from timings, memory and `generated` timestamps, a rerun changes only these:

- `tickdist_case3_check.py` and `hw.py` record absolute local paths (`source`, `meta.case3`,
  `python_exe`). The committed copies keep only the file name or the end of the path.
- `results_case3_small.json` was written by an earlier `case3.py` that did not record memory; a
  rerun adds `peak_rss_gb` and `peak_rss_method`.
- The `export` block of `storm/storm_at_scale.json` loses `symnaive_imc_bytes`, which an earlier
  version of `export_case3_storm.py` wrote.
- The `provenance` blocks of `equiv_full.json` and `synth_grid.json` hold the md5 of each source
  file the script imports, taken over the file as checked out. The committed md5s are those of the
  LF form that Git stores; a checkout with CRLF line endings records different ones (see
  Provenance below).

### Sec. IV-C: the warehouse scalability run (run from `warehouse/`)

Sec. IV-C cites |Σ|, |R|, the time and memory ratios, the identical certified interval and the
Storm check, and the abstract and the conclusion repeat some of them.
The table lists every number of the run, including those the paper does not print.

| Number | Command | Written to (fields) | Runtime |
|---|---|---|---|
| \|Σ\| = 381·7·161·8·2 = 6,870,192; H = 280 | `python case3.py full` | `results_case3_full.json` (`sigma`, `H`) | 2 min |
| the factors 381, 7, 161, 8, 2 | `python model_facts.py` | `model_facts.json` (`sigma_factorisation`) | 1 s |
| seven interval leaves, Clopper–Pearson at confidence 1 − 0.05/7 from illustrative trial counts (`COUNTS` in `case3.py`) | `python case3.py table` (prints them); `python case3.py full` | `results_case3_full.json` (`intervals`); `model_facts.json` (`leaves`) | 1 s |
| BFS reaches \|R\| = 671,532 states | `python case3.py full` | `results_case3_full.json` (`R`) | 2 min |
| BFS 40.2 s, robust VI 43 / 48 s per bound, 1.57 GB peak (medians of three runs) | `python timed_pipeline.py td_dropin` three times, then `python report.py` | `timed_runs.jsonl` (`t_bfs_s`, `t_vi_lower_s`, `t_vi_upper_s`, `peak_rss_gb_after_two_vi`) | 2.5 min per run |
| BFS with the hand-compiled successor function: 8 s (median 7.8 s), against 40 s with the compilation | `python timed_pipeline.py hand` three times, then `python report.py` | `timed_runs.jsonl` (`t_bfs_s` of mode `hand`) | 2 min per run |
| materialized IMC over Σ: 24.5 s build, 425 / 467 s per bound, 5.5 GB (one run) | `python case3.py materialize-vi` | `results_case3_materialized_vi.json` | 15.5 min, 5.5 GB RAM |
| the compilation matches the hand-compiled successor function edge for edge on all of R | `python equiv_full.py` | `equiv_full.json` (`n_states_checked` = 671,532, `n_states_mismatch` = 0, `tickdist_bfs`, `tickdist_compile_imc`) | 2.5 min |
| about a seventh of the time (132 vs 916 s) and under a third of the memory (1.57 vs 5.5 GB); each robust VI sweep about 10 times faster (43 vs 425 s, 48 vs 467 s) | `python report.py` | derived from `results_case3_materialized_vi.json` and `timed_runs.jsonl` | instant |
| both return the bit-identical certified interval 0.26794 / 0.97436 | the three commands above, then `python report.py` | `lower`, `upper` in each file (`report.py` compares the materialized file with `results_case3_full.json`); `lower_bit_identical`, `upper_bit_identical` in `timed_runs.jsonl` compare with this interval, which `timed_pipeline.py` holds as constants | |
| Storm runs on the reachable fragment, a 182 MB file | `cd storm && python export_case3_storm.py --variant full` | `storm/build_storm/case3_export.json` (`drn_imc_bytes`), copied into `storm/storm_at_scale.json` (`export`) by the next step | 2.5 min |
| Storm matches both endpoints to its 10 printed digits in 11.5 and 10.8 s at 450 MB | `cd storm && python run_storm_at_scale.py --only run1` | `storm/storm_at_scale.json` (`run1_pruned_interval_mc`: `value`, `storm_wallclock_s`, `peak_memory_mb`; `validation`) | 25 s |
| the 128 corners of Θ(T) attain [0.356, 0.963]; the interval is conservative by at most 0.088 and 0.011 | `python corners.py` (reads `results_case3_full.json`) | `results_case3_corners.json` (`corner_min`, `corner_max`, `slack_lower`, `slack_upper`) | 17.5 min |
| Intel Core Ultra 9 285K, 64 GB | `python hw.py` (Windows only) | `hw.json` | 2 s |

Notes on the warehouse run:

- **Timing protocol.** `timed_pipeline.py` has three modes. `td_dropin` is the BFS loop of
  `case3.build` with the compilation as the successor function, and it gives the timings above.
  `hand` uses the hand-compiled successor function instead, and `td_compile` uses the library's
  own `tickdist.compile_imc`. The paper's three `td_dropin` runs were interleaved with the other
  two modes, each in a fresh process:
  `for i in 1 2 3; do for m in hand td_dropin td_compile; do python timed_pipeline.py $m; done; done`
  (about 21 min). `timed_load.log` records the order and the system load before each run. Each
  run appends one line to `timed_runs.jsonl`, and `report.py` takes the median of the last three
  runs per mode (`--last N` to change).
- **Which timings are which.** `results_case3_full.json` also holds timings (BFS 8.2 s, VI 43.4 /
  49.1 s). They come from one run of the hand-compiled pipeline, not from the compilation, and
  they are not the timings above.
- **Memory.** The 1.57 GB is the process peak after both robust VI sweeps, which is where the
  5.5 GB baseline is measured too. `results_case3_full.json` also reports 1.57 GB. That number is
  a different measurement: the hand-compiled pipeline, read after an extra midpoint sweep. Peak
  memory is psutil's peak working set, which exists only on Windows. On other systems
  `timed_pipeline.py` records NaN. `case3.py` falls back to `resource.ru_maxrss` there, which makes
  its `peak_rss_gb` 1000x too small on Linux, as its `peak_rss_method` field says.
- **A seventh of the time.** 916 s is 24.5 + 425.2 + 466.7 s. The 132 s is the `td_dropin`
  median of BFS plus both VI sweeps, and 916 s is 6.9x that. It leaves out the 1.1 s array fill,
  which the 24.5 s includes; counting the fill on both sides also gives 6.9x. The memory ratio is
  5.5 / 1.57 = 3.5x. The baseline is a single run from an earlier session on the same host.
  `report.py` divides whatever is in `results_case3_materialized_vi.json` by the median in
  `timed_runs.jsonl`. If you rerun only one side, the ratio mixes two sessions and moves with the
  load: two reruns of one side on a busy machine printed 5.96x and 8.17x. To compare like for like,
  rerun both on an idle machine.
- **The edge-for-edge check.** `python tickdist_case3_check.py full` is a quicker check (a few
  seconds). It covers all reachable states of an 11x11 variant, plus 8,000 states of the full model,
  and writes `results_tickdist_case3.json`. The claim about all of R rests on `equiv_full.py`.
  `tickdist_case3_check.py` compares with the committed `results_case3_small.json`, so it runs
  from a fresh clone. `python case3.py small` rewrites that file and also writes PRISM files,
  which git ignores and the check does not need.
- **The corners.** Every corner of the leaf box is a leaf assignment, so the range the 128 corners
  attain, [0.356, 0.963], lies inside the certified interval, and the interval is conservative by
  at most 0.088 and 0.011. The corner range need not be the exact range over Θ(T): over 280 ticks
  a leaf's value enters a path once for every tick in which that leaf fires, so the probability
  need not be multi-affine in the leaf values, and its minimum or maximum over Θ(T) may lie inside
  the box.
- **Library self-tests.** `python ../ibt/tickdist.py test` runs the library's unit tests and
  writes `ibt/results_tickdist.json` (ignored by git).

### The Storm check

`export_case3_storm.py --variant full` runs `case3.build` and writes the reachable chain as an
interval-valued Storm DRN file, `storm/build_storm/case3_pruned_imc.drn` (182,480,807 bytes),
together with the property file `case3.props.pctl`, which holds `P=? [ F<=280 "goal" ]`. The DRN
comes from the hand-compiled successor function, which `equiv_full.py` shows gives the same chain
as the compilation. `--skip-vi` saves about 105 s, but it also leaves the reference bounds out of
the `export` block.

`run_storm_at_scale.py --only run1` then runs, once with `minimize` and once with `maximize`:

```bash
docker run --rm --name stormscale<pid>_<n> -v <repo>/warehouse/storm/build_storm:/data movesrwth/storm:stable \
  storm --explicit-drn /data/case3_pruned_imc.drn --prop /data/case3.props.pctl \
        --io:uncertainty-resolution minimize --timemem --resources:timeout 1800
```

If you type this command yourself in Git Bash on Windows, put `MSYS_NO_PATHCONV=1` in front of
it. Otherwise Git Bash rewrites `/data/...` into a Windows path and Storm cannot find the files.
The script does not need this, because it starts Docker directly.

The 11.5 and 10.8 s in the table are Storm's own reported wallclock (`storm_wallclock_s`). Wall
time including Docker start-up is in `wall_clock_s_incl_docker`. Storm prints 10 significant
digits, so the absolute errors near 3e-11 in `validation` mean the two agree. Each is a single
run. The errors are taken against the paper's interval, which `run_storm_at_scale.py` holds as
constants; they equal `lower` and `upper` in `results_case3_full.json` bit for bit.
`--only run1` recomputes only the `run1` entries of `validation`. The entries for the other
groups are carried over from the committed file.

The script merges its results into the existing `storm_at_scale.json`. That file also keeps other
Storm experiments made while the paper was prepared: parameter lifting, a symbolic PRISM encoding
and `probe_tightness`. The corner sampling in `probe_tightness` matches `results_case3_corners.json`
to 10 digits. `run1` backs the Storm check (11.5 + 10.8 s, 450 MB). The paper also cites, from the same file,
`run3_symbolic_interval_mc`: Storm builds the 671,532 reachable states from the 164-command module
`case3_sym_imc.prism` and returns both bounds in 9.2 + 9.1 s by its own clock, at 470 MB. That part of
`--only run3i` reruns from the files `export_case3_storm.py --variant full` writes. The other groups
(the full-Σ part of `run3i`, `run2`, `probe`) are experiments the paper does not use; some need
property and model files that no script here writes.

### Sec. V: delivery (run from `delivery/`)

`python synth_case.py` writes `results_synth.json` in about 75 s, and most rows below are fields
of that file. A rerun reproduces every field exactly except the two timings, `model.verify_s` and
`simulate_s`.

| Paper | Command | Written to (fields) | Runtime |
|---|---|---|---|
| shelf cell 0, station cell 12, zone cells 5–7, two orders, H = 80 | `python synth_case.py` | `results_synth.json` (`mission`) | 75 s |
| \|Σ\| = 78 | `python synth_case.py` | `model.sigma` | |
| \|R\| = 41 reachable states | `python synth_case.py` | `model.R` | |
| certified interval [0.854, 1 − 4e-9] | `python synth_case.py` | `certificate_H` | |
| returned in 6 ms | `python synth_case.py` | `model.verify_s` (one run of compilation and both VI sweeps; varies by host and run) | |
| the iterates give the certified interval for every deadline H = k ≤ 80 (the band in Fig. 3(c)) | `python synth_case.py` | `Vlo`, `Vhi` (81 entries each) | |
| 20,000 episodes per strategy, fixed seeds; 760,000 in all (abstract) | `python synth_case.py` | `N`, `seeds`, `episodes_total` | |
| 38 strategies: 16 corners, 20 interior draws, A, W | `python synth_case.py`; `python report.py` counts them | `rows` | |
| largest excursion outside the certified interval 0.004, by A | `python report.py` | derived from `rows[*].emp`, `Vlo`, `Vhi` | instant |
| 95% DKW margin 0.0096 | `python synth_case.py` | `dkw_95` | |
| leaf assignments' exact values at H = 80 span 0.967–1.000 | `python synth_case.py` | `fixed.range_H` | |
| largest gap between simulated and exact curves 0.0100 over all H | `python synth_case.py` | `fixed.max_cdf_gap` | |
| χ²₃₆ = 33.2, p-value 0.60 | `python synth_case.py` | `fixed.chi2_H`, `fixed.dof`, `fixed.chi2_p` | |
| A's exact value equals the lower iterate for every H to within 2e-15 | `python synth_case.py` | `adversary.pred_equals_floor_maxdiff` | |
| 0.855 of A's episodes deliver within H = 80 | `python synth_case.py` | `adversary.emp_H` | |
| W delivers 0.937 (0.936 exact) | `python synth_case.py` | `world.emp_H`, `world.pred_H` | |
| a 40-step chain | `python report.py` | first deadline with `Vhi[t] > 0` | instant |
| delivery is nondecreasing in grasp, handover and move; 0.967 is the minimum over Θ(T) | `python synth_grid.py` | `synth_grid.json` (`per_leaf_monotonicity_on_grid`, `min`, `argmin`) | 8 s |
| a dense sweep of the detector, other leaves at their lower ends, finds the minimum at 0.20, a corner | `python synth_person_sweep.py` | `synth_person_sweep.json` (`sweeps.others_low.argmin_person`) | 7 s |
| an SBT that pools W's detector rates into 0.35 predicts 0.998 | `python synth_case.py` | `world.pooled_sbt_H` | |
| the 0.11 gap between the certificate and that minimum | `python report.py` | `fixed.range_H[0] − certificate_H[0]`; also `synth_grid.json` (`gap_floor_to_grid_min`) | instant |
| Fig. 3(c) | `python fig_synth.py` (reads `results_synth.json`) | `fig_synth.pdf` (not committed; the axis labels are set in the paper's LaTeX) | 1 s |

The mission map of Fig. 3(a) is drawn in the paper's LaTeX; its cells are those of
`synth_case.py`: the shelf at cell 0, the station at cell `L`, the zone `XING`. The tree of
Fig. 3(b) is `build()` in `synth_case.py`, and its leaf intervals are `BOX`.

## Provenance

This repository was assembled from the authors' working tree, where `ibt/tickdist.py` lived in
`simulation/case4/`, `warehouse/` in `simulation/case3/`, `warehouse/storm/` in
`simulation/external/`, and `delivery/` in `simulation/synth/`. The equivalence check, the timing
script, the grid and the detector sweep come from the authors' experiment folder. `case3` in file
names, and in the header line that `export_case3_storm.py` writes into the Storm files, is the
authors' working name for the warehouse model.

The code here is the code that produced the results, with three kinds of later edit:

- **Paths.** Imports and file paths were changed for this layout (for example line 49 of
  `synth_case.py` and lines 32 and 268 of `tickdist_case3_check.py`).
- **The delivery deadline.** It moved from H = 70 to H = 80, on line 56 of `synth_case.py`, here
  and in the authors' tree alike, and the delivery result files were regenerated in this
  repository. At the same time `fig_synth.py` was changed to stack its end labels so that they do
  not overlap.
- **Text, for this release.** Comments, docstrings, usage texts, error messages, test labels and
  the descriptive strings that the scripts write (`what`, `command` and notes) were edited, so
  that they use the paper's terms and its section and figure numbers and give the file locations
  of this repository. The self-test of `ibt/tickdist.py` also no longer writes two blocks of
  development notes into `results_tickdist.json` (ignored by git), and one expected value in its
  check of the Fig. 1(c) table now holds the values the paper prints, which round the certified
  bounds outward (lower bounds down, upper bounds up).

Apart from the deadline, no algorithm, model constant or seed was changed, and none of the edits
changes which fields a script writes to a committed result file. In the committed result files,
the `command` and path strings were updated to this layout, the summary prose in
`storm_at_scale.json` was replaced by the text `run_storm_at_scale.py` now writes, and a
hand-written summary block that no script here produces (`tightness_comparison`) was removed. No
number in any result file was changed by these edits.

Because these edits change the files, the `provenance` blocks of `synth_grid.json` and
`equiv_full.json` are refreshed in this release: both scripts are rerun on the files published
here, so each block records the md5 of the files its script imports (`tickdist.py` and
`synth_case.py` for the grid; `case3.py`, `tickdist.py` and `tickdist_case3_check.py` for the
equivalence check). Each md5 is taken over the file with LF line endings, the form Git stores
and `git cat-file -p HEAD:<path> | md5sum` prints. A checkout with CRLF line endings (Git for
Windows with `core.autocrlf=true`) gives a different md5. No other result file records an md5.
