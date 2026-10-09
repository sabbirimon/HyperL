#!/usr/bin/env python3
"""Measure reviewed local CPU paths; never downloads code or selects a GPU."""
import argparse
from array import array
import ctypes as c
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from hyperl import NativeCpu, Program, Step, f32
from hyperl.native import _Vector, _Step, _Cancel


def measure(function, rounds, repeats):
    for _ in range(3):
        function()
    samples = []
    for _ in range(rounds):
        start = time.perf_counter_ns()
        for _ in range(repeats):
            function()
        samples.append((time.perf_counter_ns() - start) / repeats / 1000)
    return {'median_us': statistics.median(samples),
            'p95_batch_mean_us': sorted(samples)[math.ceil(len(samples) * .95) - 1],
            'batch_means_us': samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=9)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if not 5 <= args.rounds <= 100 or not 1 <= args.repeats <= 1000:
        parser.error('rounds must be 5..100 and repeats 1..1000')
    import numpy as np
    cpu = NativeCpu(args.library)
    if not args.reference.is_absolute() or not args.reference.is_file():
        parser.error('select an existing absolute reviewed benchmark library')
    baseline = c.CDLL(str(args.reference.resolve()))
    baseline.hl_benchmark_weighted_relu.argtypes = [c.POINTER(c.c_float)] * 3 + [c.c_size_t]
    baseline.hl_benchmark_weighted_relu.restype = None
    program = Program(('x', 'w'), (Step('p', 'multiply', ('x', 'w')),
                                    Step('y', 'relu', ('p',))), 'y')
    rows = []
    for length in (128, 4096, 65536):
        inputs = {'x': f32([(i % 17 - 8) * .25 for i in range(length)]),
                  'w': f32([(i % 9 - 4) * .5 for i in range(length)])}
        nx, nw = (np.array(inputs[name], dtype=np.float32) for name in ('x', 'w'))
        expected = np.maximum(np.multiply(nx, nw), np.float32(0))
        owned = [array('f', inputs[name]) for name in ('x', 'w')]
        buffers = [(c.c_float * length).from_buffer(data) for data in owned]
        vectors = (_Vector * 2)(*[_Vector(buf, length) for buf in buffers])
        steps = (_Step * 2)(_Step(2, 0, 1), _Step(3, 2, 0))
        output = array('f', [0]) * length
        target = (c.c_float * length).from_buffer(output)
        count = c.c_size_t(0)
        callback = _Cancel(0)

        def native_abi():
            status = cpu._lib.hl_execute(vectors, 2, steps, 2, 3, target, length,
                                         c.byref(count), callback, None)
            if status != 0 or count.value != length:
                raise RuntimeError('native benchmark failed')

        def kernel_only():
            baseline.hl_benchmark_weighted_relu(buffers[0], buffers[1], target, length)

        functions = {
            'python_api': lambda: cpu.execute(program, inputs),
            'native_abi': native_abi,
            'numpy_pipeline': lambda: np.maximum(np.multiply(nx, nw), np.float32(0)),
            'preallocated_naive_c_kernel': kernel_only,
        }
        for name, function in functions.items():
            result = function()
            observed = result if name in ('python_api', 'numpy_pipeline') else output
            np.testing.assert_array_equal(np.array(observed, dtype=np.float32), expected)
        rows.append({'elements': length, 'correctness': 'exact finite f32 outputs',
                     'paths': {name: measure(fn, args.rounds, args.repeats)
                               for name, fn in functions.items()}})
    values = f32([16777216, 1, -16777216])
    ordered = Program(('x',), (Step('s', 'sum', ('x',)),), 's')
    hardware = {'system': platform.system(), 'machine': platform.machine()}
    if sys.platform == 'darwin':
        hardware['os_version'] = subprocess.check_output(['sw_vers', '-productVersion'], text=True).strip()
        brand = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True,
                               capture_output=True, check=False)
        hardware['cpu'] = brand.stdout.strip() if brand.returncode == 0 else 'unavailable in current permissions'
    report = {
        'format': 'hyperl-cpu-benchmark/1', 'hardware': hardware,
        'python': platform.python_version(), 'numpy': np.__version__,
        'native_sha256': hashlib.sha256(args.library.read_bytes()).hexdigest(),
        'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        'warmups': 3, 'rounds': args.rounds, 'calls_per_round': args.repeats,
        'workload': 'two-step finite f32 multiply then relu; no GPU/model/transfer',
        'cost_scopes': {
            'python_api': 'Python validation, private snapshots, ctypes setup, admitted native allocations/computation, output',
            'native_abi': 'ctypes call plus C validation, admitted allocations/computation and output copy; caller preparation outside timer',
            'numpy_pipeline': 'two NumPy ufuncs with intermediate/output allocation; input preparation outside timer',
            'preallocated_naive_c_kernel': 'ctypes call plus one C loop; input validation/admission/preallocation excluded',
        },
        'rows': rows,
        'precision_example': {'input': list(values), 'ordered_sum_v1': cpu.execute(ordered, {'x': values})[0],
                              'precise_sum_1': cpu.precise_sum(values)[0], 'math_fsum': math.fsum(values)},
        'limits': ['single host, bounded workload, sequential path order, no application-speedup claim',
                   'p95 is across batch means, not individual-call tail latency',
                   'different cost scopes; do not present ratios as equivalent end-to-end workloads'],
    }
    with args.output.open('x', encoding='utf-8') as output_file:
        json.dump(report, output_file, indent=2, allow_nan=False)
        output_file.write('\n')
    print(json.dumps({'output': str(args.output), 'rows': len(rows), 'correctness': 'passed'}))


if __name__ == '__main__':
    main()
