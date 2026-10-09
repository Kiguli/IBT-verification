"""Shared setup for equiv_full.py, timed_pipeline.py, model_facts.py and hw.py.

Puts ../ibt (tickdist.py) and this folder (case3.py, tickdist_case3_check.py) on sys.path, and
provides the peak-memory reading and the provenance block those scripts record.

The paper's runs used an earlier version of this module that imported md5-checked copies of the
same files from another folder; provenance() now records the md5 of the files actually
imported here. Its output is written to this folder (HERE).
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IBT = os.path.join(os.path.dirname(HERE), 'ibt')
sys.path.insert(0, IBT)
sys.path.insert(0, HERE)


def md5(path):
    with open(path, 'rb') as fh:
        return hashlib.md5(fh.read()).hexdigest()


def provenance():
    """md5 of the source files the experiments import (as checked out, so line endings count)."""
    files = {'case3.py': os.path.join(HERE, 'case3.py'),
             'tickdist.py': os.path.join(IBT, 'tickdist.py'),
             'tickdist_case3_check.py': os.path.join(HERE, 'tickdist_case3_check.py')}
    return {name: {'md5': md5(path)} for name, path in files.items()}


_warned = []


def peak_wset_mb():
    """Process-wide peak working set in MB: psutil peak_wset (bytes) / 1e6.

    This is the branch case3._peak_rss_mb takes on Windows, where every memory figure in the paper
    was measured. peak_wset exists only on Windows; elsewhere this returns NaN (and says so once on
    stderr) rather than a number measured another way under the same label.
    """
    import psutil
    info = psutil.Process().memory_info()
    if hasattr(info, 'peak_wset'):
        return info.peak_wset / 1e6
    if not _warned:
        _warned.append(True)
        print('exp_common: peak_wset is Windows-only; peak memory is reported as NaN',
              file=sys.stderr)
    return float('nan')
