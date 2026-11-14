# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
"""Selection and configuration checks; the fixture never claims GPU execution."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from hyperl import BackendUnavailable, LlamaSettings, open_compute, open_model
from hyperl.torch_backend import _resolve_device


class SdkContractTests(unittest.TestCase):
    def test_rocm_and_cuda_do_not_accept_each_others_runtime(self):
        devices = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)
        torch = SimpleNamespace(version=SimpleNamespace(hip="7", cuda=None), cuda=devices)
        self.assertEqual(_resolve_device(torch, "rocm:1"), "cuda:1")
        with self.assertRaises(BackendUnavailable):
            _resolve_device(torch, "cuda:1")
        torch.version = SimpleNamespace(hip=None, cuda="12")
        self.assertEqual(_resolve_device(torch, "cuda:0"), "cuda:0")
        with self.assertRaises(BackendUnavailable):
            _resolve_device(torch, "rocm:0")

    def test_missing_or_out_of_range_accelerator_cannot_resolve_cpu(self):
        torch = SimpleNamespace(xpu=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1))
        self.assertEqual(_resolve_device(torch, "xpu:0"), "xpu:0")
        with self.assertRaises(BackendUnavailable):
            _resolve_device(torch, "xpu:1")
        del torch.xpu
        with self.assertRaises(BackendUnavailable):
            _resolve_device(torch, "xpu:0")

    def test_compute_requires_exactly_one_cpu_reference(self):
        with self.assertRaises(ValueError):
            open_compute()
        with self.assertRaises(ValueError):
            open_compute(library="/unreviewed", bundled=True)

    def test_partial_offload_rejects_cpu_disguise_and_out_of_range(self):
        artifact = ("/unused", "0" * 64, "/unused", "0" * 64)
        LlamaSettings(*artifact, backend="GPU", devices=("Metal",), gpu_layers=3).validate()
        for options in (dict(gpu_layers=1), dict(backend="GPU", devices=("CPU",), gpu_layers=1),
                        dict(backend="GPU", devices=("Metal",), gpu_layers=0),
                        dict(backend="RPC", rpc=("127.0.0.1:9999",), gpu_layers=3)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                LlamaSettings(*artifact, **options).validate()

    def test_cancelled_model_never_touches_artifacts(self):
        settings = LlamaSettings("/does-not-exist", "0" * 64, "/does-not-exist", "0" * 64)
        with self.assertRaises(InterruptedError), open_model(settings, cancelled=lambda: True):
            self.fail("cancelled model session opened")


if __name__ == "__main__":
    unittest.main()
