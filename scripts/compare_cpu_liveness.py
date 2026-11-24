#!/usr/bin/env python3
"""Compare two explicitly selected CPU libraries; no downloads, GPU or network."""
# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
import argparse
import ctypes as c
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import time


class Vector(c.Structure):
    _fields_ = [('data', c.POINTER(c.c_float)), ('length', c.c_size_t)]


class Step(c.Structure):
    _fields_ = [('operation', c.c_int), ('a', c.c_size_t), ('b', c.c_size_t)]


Cancel = c.CFUNCTYPE(c.c_int, c.c_void_p)


class Runtime:
    def __init__(self, path):
        if not path.is_absolute() or not path.is_file():
            raise ValueError('Select an existing absolute reviewed library')
        self.lib = c.CDLL(str(path))
        self.lib.hl_version.restype = c.c_char_p
        if self.lib.hl_version() != b'hyperl-cpu/1':
            raise ValueError('Unsupported native contract')
        self.lib.hl_execute.argtypes = [c.POINTER(Vector), c.c_size_t, c.POINTER(Step), c.c_size_t,
                                       c.c_size_t, c.POINTER(c.c_float), c.c_size_t,
                                       c.POINTER(c.c_size_t), Cancel, c.c_void_p]
        self.lib.hl_execute.restype = c.c_int

    def call(self, case):
        inputs, steps, selected, cancel_after = case
        owned = [(c.c_float * len(v))(*v) for v in inputs]
        vectors = (Vector * len(owned))(*[Vector(v, len(v)) for v in owned])
        operations = (Step * len(steps))(*[Step(*v) for v in steps])
        output = (c.c_float * max(map(len, inputs)))(*([12345] * max(map(len, inputs))))
        length = c.c_size_t(777)
        calls = 0

        def cancel(_):
            nonlocal calls
            calls += 1
            return int(calls >= cancel_after)

        callback = Cancel(cancel) if cancel_after else Cancel(0)
        status = self.lib.hl_execute(vectors, len(vectors), operations, len(operations), selected,
                                     output, len(output), c.byref(length), callback, None)
        if status:
            if length.value != 0 or any(v != 12345 for v in output):
                raise AssertionError('Failure published partial output')
            return status, b''
        return status, c.string_at(c.addressof(output), length.value * c.sizeof(c.c_float))


def cases():
    rng = random.Random(20261029)
    for i in range(500):
        length = rng.choice([1, 3, 17, 65, 1025])
        inputs = [[rng.randint(-8, 8) * .125 for _ in range(length)] for _ in range(2)]
        steps = []
        for index in range(rng.randint(1, 12)):
            steps.append((rng.choice([1, 2, 3]), rng.randrange(index + 2), rng.randrange(index + 2)))
        selected = rng.randrange(len(steps) + 2)
        if i % 7 == 0:
            steps.append((4, selected, 0)); selected = len(steps) + 1
        if i % 17 == 0:
            inputs[1][0] = float('nan')
        if i % 19 == 0:
            inputs[0][0] = 3e38
            steps[0] = (2, 0, 0)
        yield inputs, steps, selected, 4 if i % 11 == 0 else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Evidence output already exists')
    before, after = Runtime(args.before), Runtime(args.after)
    digest = hashlib.sha256(); count = 0
    for case in cases():
        old, new = before.call(case), after.call(case)
        if old != new:
            raise AssertionError('Status/bitwise output differs at case ' + str(count))
        digest.update(bytes([new[0]]) + new[1]); count += 1
    case = ([[float(i % 17 - 8) for i in range(65536)]], [(3, i, 0) for i in range(12)], 12, 0)
    assert before.call(case) == after.call(case)
    for _ in range(3):
        before.call(case); after.call(case)
    samples = {'before': [], 'after': []}
    # Alternate order to reduce systematic warm/cold bias; includes ctypes setup.
    for round_index in range(9):
        order = [('before', before), ('after', after)]
        if round_index % 2:
            order.reverse()
        for name, runtime in order:
            started = time.perf_counter_ns()
            runtime.call(case)
            samples[name].append((time.perf_counter_ns() - started) / 1e6)
    report = {
        'format': 'hyperl-cpu-liveness-comparison/1', 'system': platform.system(), 'machine': platform.machine(),
        'cases': count, 'bitwiseOutputsAndStatusesMatch': True, 'resultDigest': digest.hexdigest(),
        'beforeSha256': hashlib.sha256(args.before.read_bytes()).hexdigest(),
        'afterSha256': hashlib.sha256(args.after.read_bytes()).hexdigest(),
        'measurement': '12 ReLU steps, 65536 elements; Python/ctypes setup, native execution and output copy',
        'warmupsPerLibrary': 3, 'samplesMs': samples,
        'medianMs': {name: statistics.median(values) for name, values in samples.items()},
        'p95Ms': {name: max(values) for name, values in samples.items()},
        'limits': ['nine samples: p95 is maximum', 'single host', 'not a kernel-only benchmark',
                   'no process RSS measurement', 'no GPU/phone/cluster speedup claim'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps({'cases': count, 'matched': True, 'medianMs': report['medianMs']}))


if __name__ == '__main__':
    main()
