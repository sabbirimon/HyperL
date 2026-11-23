import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "studio"))
from cuda_migrate import HIPIFY, migrate, perl
from cuda_kernel import interchange, lower

SOURCE = (ROOT / "examples/cuda/add.cu").read_text()


class Migration(unittest.TestCase):
    def test_complete_grammar(self):
        for op in "+-*":
            text, spec = lower(SOURCE.replace("left[i] + right[i]", f"right[i] {op} left[i]"))
            self.assertEqual(spec["operands"], [1, 0])
            self.assertIn(f"data1[id.x] {op} data0[id.x]", text)
        for src in [SOURCE + "void extra(){}", SOURCE.replace("i < count", "i > count"),
                    SOURCE.replace("threadIdx.x", "threadIdx.y"), SOURCE.replace("right[i]", "right[i+1]"),
                    "#define float int\n" + SOURCE, SOURCE.replace("left[i]", "result[i]"),
                    SOURCE.replace("unsigned int i", "unsigned int count"), "//line\\\n" + SOURCE]:
            with self.subTest(src=src), self.assertRaises(ValueError):
                lower(src)

    @unittest.skipUnless(perl(), "requires Perl 5")
    def test_pytorch_uses_the_recorded_buffer_order(self):
        for expression, expected, bindings in [
            ("right[i] - left[i]", "torch.sub(data1, data0)", [1, 0]),
            ("left[i] * left[i]", "torch.mul(data0, data0)", [0]),
        ]:
            _, spec = lower(SOURCE.replace("left[i] + right[i]", expression))
            code = interchange("pytorch", spec)
            compile(code, "kernel.py", "exec")
            self.assertEqual(spec["graphInputs"], bindings)
            self.assertIn(expected, code)
            self.assertIn("dtype == torch.float32", code)

    def test_batch_matches_upstream_including_unknown_apis(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            src = base / "input"
            (src / "dir with spaces").mkdir(parents=True)
            inputs = {"source file.cu": SOURCE, "dir with spaces/source file.cu": "void f(){cudaUnknownDeviceCall();}\n",
                      "unchanged.h": "extern int x;\n", "header.cuh": "#include <cuda_runtime.h>\n", "empty.h": ""}
            for name, data in inputs.items():
                (src / name).write_text(data)
            dest = base / "new"
            report = migrate(src, dest, "hip")
            for record in report["sources"]:
                name = record["path"]
                actual = subprocess.run([perl(), str(HIPIFY), str(src / name)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                self.assertEqual((dest / name).read_bytes(), actual.stdout)
                self.assertEqual((src / name).read_text(), inputs[name])
                self.assertEqual(record["upstreamWarnings"], int("Unknown" in inputs[name]))
            self.assertFalse(any(dest.rglob("*.prehip")))
            with self.assertRaises(ValueError):
                migrate(src, dest, "hip")
            with self.assertRaises(ValueError):
                migrate(src, src / "out", "hip")

    def test_interchange_and_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source = base / "example.cu"
            source.write_text(SOURCE)
            for target, suffix in [("opencl", "cl"), ("stablehlo", "mlir"), ("onnx-text", "onnxtxt"), ("pytorch", "py")]:
                report = migrate(source, base / target, target)
                self.assertTrue((base / target / ("kernel." + suffix)).is_file())
                self.assertFalse(report["hardwareExecuted"])
            source.write_text(SOURCE.replace("threadIdx.x", "threadIdx.y"))
            with self.assertRaises(ValueError):
                migrate(source, base / "failed", "opencl")
            self.assertFalse((base / "failed").exists())


if __name__ == "__main__":
    unittest.main()
