"""Shared setup for synth_grid.py and synth_person_sweep.py.

Puts ../ibt (tickdist.py) on sys.path and provides the provenance block synth_grid.py records.
Output is written to this folder (HERE).

The paper's runs used an earlier version of this module that imported md5-checked copies of the
same files from another folder; provenance() now records the md5 of the files actually
imported here.
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IBT = os.path.join(os.path.dirname(HERE), 'ibt')
sys.path.insert(0, IBT)


def md5(path):
    with open(path, 'rb') as fh:
        return hashlib.md5(fh.read()).hexdigest()


def provenance():
    """md5 of the source files the experiments import (as checked out, so line endings count)."""
    files = {'tickdist.py': os.path.join(IBT, 'tickdist.py'),
             'synth_case.py': os.path.join(HERE, 'synth_case.py')}
    return {name: {'md5': md5(path)} for name, path in files.items()}
