"""Run with -I after installing a platform wheel in a clean virtual environment."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import hyperl
from hyperl import NativeCpu, Program, Step, f32
from hyperl.bundled import bundled_native_path


class BundledWheelTests(unittest.TestCase):
    def test_actual_installed_native_binary_generates_and_cancels(self):
        cpu = NativeCpu.bundled()
        program = Program(('x',), (Step('y', 'relu', ('x',)),), 'y')
        self.assertEqual([0, 2, 3], list(cpu.execute(program, {'x': f32([-1, 2, 3])})))
        self.assertEqual([1], list(cpu.precise_sum(f32([16777216, 1, -16777216]))))
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            cpu.execute(program, {'x': f32([-1, 2, 3])}, cancelled=lambda: True)

    def test_binary_corruption_is_rejected_before_load(self):
        installed = Path(hyperl.__file__).parent / '_native'
        with tempfile.TemporaryDirectory(prefix='hyperl-bundle-') as folder:
            root = Path(folder) / 'bundle'
            shutil.copytree(installed, root)
            record = json.loads((root / 'manifest.json').read_text())
            library = root / record['library']
            library.write_bytes(library.read_bytes() + b'changed')
            with self.assertRaisesRegex(RuntimeError, 'digest mismatch'):
                bundled_native_path(root)

    def test_path_and_target_substitution_are_rejected(self):
        installed = Path(hyperl.__file__).parent / '_native'
        with tempfile.TemporaryDirectory(prefix='hyperl-bundle-') as folder:
            root = Path(folder) / 'bundle'
            shutil.copytree(installed, root)
            record = json.loads((root / 'manifest.json').read_text())
            for field, value, error in [('library', '../outside.so', 'filename'),
                                         ('machine', 'wrong-architecture', 'target')]:
                bad = {**record, field: value}
                (root / 'manifest.json').write_text(json.dumps(bad))
                with self.assertRaisesRegex(RuntimeError, error):
                    bundled_native_path(root)


if __name__ == '__main__':
    unittest.main()
