# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
"""Measure a compiled F32 matrix product, including input upload and output readback."""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from hyperl.ir import DType, Graph, Type


def sample(call, expected, rounds):
    import numpy as np
    result = call()
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=3e-4)
    times = []
    for _ in range(rounds):
        t = time.perf_counter_ns()
        result = call()
        times.append((time.perf_counter_ns() - t) / 1e6)
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=3e-4)
    return {"median_ms": statistics.median(times), "samples_ms": times}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--rounds", type=int, default=5)
    args = p.parse_args()
    if not 1 <= args.size <= 2048 or not 1 <= args.rounds <= 100:
        p.error("size or sample count out of bounds")
    if args.device == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible or "," in visible:
            p.error("set CUDA_VISIBLE_DEVICES to one physical NVIDIA GPU")
    import jax
    import numpy as np
    cuda = args.device == "cuda"
    device = jax.devices("gpu" if cuda else "cpu")[0]
    n = args.size
    rng = np.random.default_rng(12)
    x, y = (rng.uniform(-1,1,(n,n)).astype(np.float32) for _ in range(2))
    expected = (x.astype(np.float64) @ y.astype(np.float64)).astype(np.float32)
    g = Graph()
    a,b = [g.add_input(name,Type(DType.F32,(n,n))) for name in ["a","b"]]
    g.set_outputs([g.matmul(a,b)])
    start = time.perf_counter_ns()
    run = g.lower().compile(device)
    report = {"device": device.device_kind, "platform_version": device.client.platform_version,
              "shape": [n,n,n], "dtype": "f32", "jax": jax.__version__,
              "scope": "warm host input to host output; compilation separate",
              "compile_ms": (time.perf_counter_ns()-start)/1e6}
    # device_get waits for completion and copies GPU results back to the host.
    report["stablehlo_xla"] = sample(lambda: np.asarray(run(x,y)[0]), expected, args.rounds)
    if cuda:
        import torch
        if not torch.version.cuda or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            p.error("one visible CUDA device with a CUDA-enabled PyTorch build required")
        torch.backends.cuda.matmul.allow_tf32 = False
        report.update(torch=str(torch.__version__), cuda=torch.version.cuda, torch_device=torch.cuda.get_device_name(0))
        report["torch_cuda"] = sample(lambda: (torch.from_numpy(x).cuda() @ torch.from_numpy(y).cuda()).cpu().numpy(), expected, args.rounds)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
