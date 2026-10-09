"""Hardware / software of the benchmark host (Windows only). Writes hw.json in this folder."""
import json
import os
import platform
import subprocess
import sys
import time

import numpy
import psutil
import scipy

import exp_common as ec


def cim(cls, props):
    cmd = ('Get-CimInstance %s | Select-Object %s | ConvertTo-Json -Compress' % (cls, ','.join(props)))
    out = subprocess.run(['powershell', '-NoProfile', '-Command', cmd], capture_output=True, text=True).stdout
    return json.loads(out)


cpu = cim('Win32_Processor', ['Name', 'NumberOfCores', 'NumberOfLogicalProcessors', 'MaxClockSpeed'])
cs = cim('Win32_ComputerSystem', ['TotalPhysicalMemory', 'Manufacturer', 'Model'])
os_ = cim('Win32_OperatingSystem', ['Caption', 'Version', 'BuildNumber', 'OSArchitecture'])
mem = cim('Win32_PhysicalMemory', ['Capacity', 'Speed', 'SMBIOSMemoryType'])
mem = mem if isinstance(mem, list) else [mem]
res = {
    'cpu': cpu['Name'], 'physical_cores': cpu['NumberOfCores'],
    'logical_processors': cpu['NumberOfLogicalProcessors'],
    'cpu_note': 'NumberOfCores == NumberOfLogicalProcessors, i.e. one thread per core (no SMT)',
    'max_clock_mhz_reported': cpu['MaxClockSpeed'],
    'ram_bytes': int(cs['TotalPhysicalMemory']), 'ram_gb_decimal': round(int(cs['TotalPhysicalMemory']) / 1e9, 1),
    'ram_modules': [{'capacity_gib': int(m['Capacity']) / 2 ** 30, 'speed_mt_s': m['Speed'],
                     'smbios_type': m['SMBIOSMemoryType']} for m in mem],
    'ram_note': 'SMBIOS memory type 34 = DDR5',
    'machine': '%s %s' % (cs['Manufacturer'], cs['Model']),
    'os': '%s %s (build %s, %s)' % (os_['Caption'], os_['Version'], os_['BuildNumber'], os_['OSArchitecture']),
    'python': sys.version.split()[0], 'python_full': sys.version,
    'numpy': numpy.__version__, 'scipy': scipy.__version__, 'psutil': psutil.__version__,
    'platform': platform.platform(),
    'python_exe': sys.executable,
    'single_threaded_note': 'The BFS loops are pure Python (one core); case3.robust_vi is numpy '
                            '(argsort/cumsum, no BLAS-heavy kernels).',
    'command': 'python hw.py',
    'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
}
with open(os.path.join(ec.HERE, 'hw.json'), 'w') as fh:
    json.dump(res, fh, indent=1)
print(json.dumps(res, indent=1))
