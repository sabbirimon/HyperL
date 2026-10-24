"""Actual same-program JVM/Python/C conformance and deterministic NumPy oracle."""
import argparse
from array import array
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hyperl import NativeCpu, Program, Step, f32

CLI = None
LIBRARY = None

def cases():
    def relu(name='x', value=None):
        return ({'format': 'hyperl/1', 'inputs': [name],
                 'instructions': [{'output': 'p', 'operation': 'relu', 'inputs': [name]}],
                 'output': 'p'}, {name: [-1, 2, 3] if value is None else value})
    values = []
    for name, valid in [('x', True), ('X' * 32, True), ('_tmp', False),
                        ('X' * 33, False), ('x\n', False), ('1x', False), ('é', False)]:
        program, inputs = relu(name)
        values.append((name, program, inputs, valid))
    program, inputs = relu()
    program.pop('format')
    values.append(('default-format', program, inputs, True))
    program, inputs = relu()
    program['inputs'].append('x')
    values.append(('duplicate-input', program, inputs, False))
    program, inputs = relu()
    program['instructions'][0]['inputs'] = ['future']
    values.append(('forward-reference', program, inputs, False))
    program, inputs = relu()
    inputs['extra'] = [1]
    values.append(('extra-input-data', program, inputs, False))
    program, inputs = relu()
    program['instructions'][0]['unused'] = True
    values.append(('unknown-step-field', program, inputs, False))
    program, inputs = relu(value=[1e40])
    values.append(('nonfinite-after-f32', program, inputs, False))
    for data, valid in [([16777216, 1, -16777216], True),
                        ([3.4028234663852886e38] * 2 + [-3.4028234663852886e38], False)]:
        program, inputs = relu(value=data)
        program['instructions'][0]['operation'] = 'sum'
        values.append(('ordered-sum-' + str(valid), program, inputs, valid))
    program, inputs = relu()
    program['instructions'] = [
        {'output': f'p{i}', 'operation': 'relu', 'inputs': ['x' if i == 0 else f'p{i-1}']}
        for i in range(65)]
    program['output'] = 'p64'
    values.append(('step-count-bound', program, inputs, False))
    return values

class ConformanceTests(unittest.TestCase):
    def test_same_json_contract_through_real_jvm_and_c(self):
        self.assertIsNotNone(CLI, 'supply an explicit built CLI')
        cpu = NativeCpu(LIBRARY)
        with tempfile.TemporaryDirectory(prefix='hyperl-conformance-') as folder:
            root = Path(folder)
            for label, program, inputs, valid in cases():
                with self.subTest(case=label):
                    (root / 'program.json').write_text(json.dumps(program, allow_nan=False))
                    (root / 'inputs.json').write_text(json.dumps(inputs, allow_nan=False))
                    native = None
                    try:
                        native = cpu.execute(Program.from_dict(program), {n: f32(v) for n, v in inputs.items()})
                    except (ValueError, RuntimeError, MemoryError):
                        pass
                    command = [CLI, 'run', str(root / 'program.json'), str(root / 'inputs.json')]
                    if sys.platform == 'win32':
                        command = ['cmd.exe', '/d', '/c'] + command
                    result = subprocess.run(command,
                                            text=True, capture_output=True, timeout=15)
                    self.assertEqual(valid, native is not None, 'Python/C acceptance')
                    self.assertEqual(valid, result.returncode == 0, 'JVM acceptance: ' + result.stderr)
                    if valid:
                        self.assertEqual(list(native), list(f32(json.loads(result.stdout))))

    def test_seeded_dags_against_numpy_ordered_f32_oracle(self):
        import numpy as np
        cpu = NativeCpu(LIBRARY)
        rng = random.Random(20261008)
        for case in range(120):
            with self.subTest(case=case):
                length = rng.choice([1, 3, 17, 1025])
                inputs = {n: f32([rng.uniform(-2, 2) for _ in range(length)]) for n in ('x', 'y')}
                oracle = {n: np.array(v, dtype=np.float32) for n, v in inputs.items()}
                steps = []
                vectors = ['x', 'y']
                for i in range(rng.randint(1, 8)):
                    op = rng.choice(['add', 'multiply', 'relu'])
                    operands = tuple(rng.choice(vectors) for _ in range(2 if op != 'relu' else 1))
                    name = f'p{i}'
                    if op == 'add':
                        oracle[name] = np.add(oracle[operands[0]], oracle[operands[1]], dtype=np.float32)
                    elif op == 'multiply':
                        oracle[name] = np.multiply(oracle[operands[0]], oracle[operands[1]], dtype=np.float32)
                    else:
                        oracle[name] = np.maximum(np.float32(0), oracle[operands[0]])
                    steps.append(Step(name, op, operands))
                    vectors.append(name)
                if case % 2 == 0:
                    total = np.float32(0)
                    for v in oracle[name]:
                        total = np.float32(total + v)
                    steps.append(Step('total', 'sum', (name,)))
                    name = 'total'
                    oracle[name] = np.array([total], dtype=np.float32)
                program = Program(('x', 'y'), tuple(steps), name)
                result = np.array(cpu.execute(program, inputs), dtype=np.float32)
                # ReLU may normalize signed zero; all nonzero values are bit exact.
                np.testing.assert_array_equal(result, oracle[name])

    def test_precise_sum_matches_jvm_and_recovers_cancellation(self):
        cpu = NativeCpu(LIBRARY)
        with tempfile.TemporaryDirectory(prefix='hyperl-precise-') as folder:
            path = Path(folder) / 'inputs.json'
            for values, expected in [([16777216, 1, -16777216], 1.0),
                                     ([3.4028234663852886e38] * 2 + [-3.4028234663852886e38], 3.4028234663852886e38)]:
                self.assertEqual(expected, cpu.precise_sum(f32(values))[0])
                path.write_text(json.dumps({'x': values}))
                command = [CLI, 'sum-precise', str(path)]
                if sys.platform == 'win32':
                    command = ['cmd.exe', '/d', '/c'] + command
                result = subprocess.run(command, text=True, capture_output=True, timeout=15)
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertEqual([expected], list(f32(json.loads(result.stdout))))
        with self.assertRaisesRegex(RuntimeError, 'nonfinite'):
            cpu.precise_sum(f32([3.4028234663852886e38] * 2))
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            cpu.precise_sum(f32([1]), cancelled=lambda: True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--library', required=True)
    parser.add_argument('--cli', required=True)
    args, rest = parser.parse_known_args()
    LIBRARY = args.library
    CLI = args.cli
    unittest.main(argv=[sys.argv[0]] + rest)
