import argparse
from array import array
import importlib.util
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hyperl import NativeCpu, Program, Step, f32, weighted_relu, residual_relu, positive_sum, LIBRARY as LIBRARY_IDS, recipe_program
from hyperl.interop import from_numpy, to_numpy, from_torch, to_torch

LIBRARY = None


class NativeTests(unittest.TestCase):
    def setUp(self):
        if LIBRARY is None:
            self.skipTest("explicit native shared library path not supplied")
        self.cpu = NativeCpu(LIBRARY)

    def test_real_recipes_and_immutable_input(self):
        x = f32([-1, 2, 3]); w = f32([2, 3, 4])
        self.assertEqual([0, 6, 12], list(weighted_relu(self.cpu, x, w)))
        self.assertEqual([-1, 2, 3], list(x))
        self.assertEqual([1, 5, 7], list(residual_relu(self.cpu, x, w)))
        self.assertEqual([5], list(positive_sum(self.cpu, x)))

    def test_existing_program_json_import(self):
        data = json.loads((Path(__file__).resolve().parents[2] / "examples/elementwise.json").read_text())
        result = self.cpu.execute(Program.from_dict(data), {"x": f32([-1, 2, 3]), "w": f32([2, 3, 4])})
        self.assertEqual([0, 6, 12], list(result))

    def test_twelve_library_programs_execute_native_cpu(self):
        expected = ([3, -1, 4], [-2, 6, 12], [0, 2, 3], [0, 6, 12], [0, 1, 3],
                    [-1, 7, 13], [0, 7, 13], [0, 6, 13], [16], [4], [5], [14])
        values = {k: f32(v) for k, v in {"x": [-1, 2, 3], "w": [2, 3, 4],
                  "y": [4, -3, 1], "bias": [1, 1, 1], "residual": [1, -1, 0]}.items()}
        for name, output in zip(LIBRARY_IDS, expected):
            with self.subTest(recipe=name):
                program = recipe_program(name)
                self.assertEqual(output, list(self.cpu.execute(program, {k: values[k] for k in program.inputs})))
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            self.cpu.execute(recipe_program("sum"), {"x": f32([3.402823466e38, 3.402823466e38, -3.402823466e38])})

    def test_overflow_before_relu_shape_and_budget(self):
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            weighted_relu(self.cpu, f32([-3.4e38]), f32([2]))
        with self.assertRaisesRegex(ValueError, "shapes"):
            weighted_relu(self.cpu, f32([1, 2]), f32([3]))
        tiny = NativeCpu(LIBRARY, budget_bytes=1)
        with self.assertRaises(MemoryError):
            positive_sum(tiny, f32([1]))

    def test_cancel_cleanup_and_callback_exception(self):
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            positive_sum(self.cpu, f32([1]), cancelled=lambda: True)
        def bad():
            raise ValueError("test cancellation failure")
        with self.assertRaisesRegex(RuntimeError, "cleanup completed"):
            positive_sum(self.cpu, f32([1]), cancelled=bad)
        def interrupt():
            raise KeyboardInterrupt()
        with self.assertRaisesRegex(RuntimeError, "cleanup completed"):
            positive_sum(self.cpu, f32([1]), cancelled=interrupt)
        self.assertEqual([2], list(positive_sum(self.cpu, f32([2]))))

    def test_retained_bound_before_copies(self):
        value = array("f", [1]) * 262144
        graph = Program(("x", "w", "z", "q"), (Step("p", "relu", ("x",)),), "p")
        with self.assertRaisesRegex(ValueError, "retained"):
            self.cpu.execute(graph, {name: value for name in graph.inputs})

    def test_numpy_roundtrip_copy_and_rejection(self):
        if importlib.util.find_spec("numpy") is None:
            self.skipTest("NumPy not installed")
        import numpy as np
        source = np.array([-1, 2, 3], dtype=np.float32)
        result = to_numpy(weighted_relu(self.cpu, from_numpy(source), f32([2, 3, 4])))
        self.assertEqual([0, 6, 12], result.tolist())
        result[0] = 99
        self.assertEqual(-1, source[0])
        for bad in (np.array([1.0], dtype=np.float64), np.ones((1, 2), dtype=np.float32), np.arange(4, dtype=np.float32)[::2], np.array([np.nan], dtype=np.float32)):
            with self.assertRaises(ValueError):
                from_numpy(bad)

    def test_torch_cpu_real_dispatch_and_roundtrip(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("PyTorch not installed")
        import torch
        from hyperl.torch_backend import TorchBackend
        x = torch.tensor([-1, 2, 3], dtype=torch.float32)
        w = torch.tensor([2, 3, 4], dtype=torch.float32)
        backend = TorchBackend(self.cpu, "cpu")
        result = weighted_relu(backend, x, w)
        self.assertEqual([0, 6, 12], result.tolist())
        self.assertEqual([0, 6, 12], list(from_torch(to_torch(f32([0, 6, 12])))))
        with self.assertRaisesRegex(ValueError, "reduction"):
            positive_sum(backend, x)
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            weighted_relu(backend, torch.tensor([-3.4e38], dtype=torch.float32), torch.tensor([2], dtype=torch.float32))
        with self.assertRaises(ValueError):
            from_torch(x.requires_grad_())

    def test_torch_cuda_real_dispatch_or_unavailable_rejection(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("PyTorch/CUDA not installed")
        import torch
        from hyperl.torch_backend import TorchBackend
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
                TorchBackend(self.cpu, "cuda:0")
            self.skipTest("CUDA unavailable; constructor rejection passed, actual dispatch unrun")
        backend = TorchBackend(self.cpu, "cuda:0")
        x = torch.tensor([-1, 2, 3], dtype=torch.float32, device="cuda:0")
        w = torch.tensor([2, 3, 4], dtype=torch.float32, device="cuda:0")
        result = weighted_relu(backend, x, w)
        self.assertEqual("cuda", result.device.type)
        self.assertEqual([0, 6, 12], result.cpu().tolist())
        with self.assertRaises(ValueError):
            from_torch(result)


class ValidationTests(unittest.TestCase):
    def test_invalid_program_and_explicit_binary_selection(self):
        with self.assertRaises(ValueError):
            NativeCpu("relative-library")
        with self.assertRaises(ValueError):
            Program(("x",), (Step("p", "relu", ("future",)),), "p")
        with self.assertRaises(ValueError):
            Program.from_dict({"format": "hyperl/1", "inputs": ["x"], "instructions": [], "output": "x", "hook": "unused"})
        with self.assertRaises(ValueError):
            f32([float("inf")])
        with self.assertRaises(TypeError):
            f32(iter([1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--library")
    options, rest = parser.parse_known_args()
    LIBRARY = options.library
    unittest.main(argv=[sys.argv[0]] + rest)
