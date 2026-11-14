# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
"""Run a bounded known-answer workload on an explicitly selected SDK backend."""
import argparse
import hashlib
import json
import platform
import re
import struct
import sys
import time

from . import BackendUnavailable, SDK_API_VERSION, __version__, f32, open_compute, weighted_relu


def verify_compute(args):
    backend = open_compute(args.target, library=args.library, bundled=args.bundled)
    size = args.elements
    x = [i % 17 - 8 for i in range(size)]
    w = [i % 7 + 1 for i in range(size)]
    expected = [max(0, a * b) for a, b in zip(x, w)]
    started = time.monotonic()
    evidence = {"target": args.target}
    if args.target == "cpu":
        result = list(weighted_relu(backend, f32(x), f32(w)))
        evidence.update(runtime="hyperl-cpu", device="cpu")
    else:
        import torch
        dev = backend.device
        result = weighted_relu(backend, torch.tensor(x, dtype=torch.float32, device=dev),
                               torch.tensor(w, dtype=torch.float32, device=dev))
        if result.device != dev:
            raise RuntimeError("result device differs from explicit selection")
        target = args.target.split(":")[0]
        runtime = getattr(torch, dev.type)
        evidence.update(runtime="pytorch", runtime_version=str(torch.__version__),
                        accelerator_version=getattr(torch.version, {"cuda": "cuda", "rocm": "hip", "xpu": "xpu"}[target], None),
                        device=str(dev), device_name=runtime.get_device_name(dev))
        result = result.cpu().tolist()
    if result != expected:
        raise RuntimeError("known-answer mismatch")
    return dict(format="hyperl-hardware-check/1", sdk=SDK_API_VERSION, package=__version__,
                os=platform.system(), architecture=platform.machine(), os_release=platform.release(),
                status="passed", workload="weighted_relu/f32", elements=size,
                output_sha256=hashlib.sha256(struct.pack(f"<{size}f", *result)).hexdigest(),
                end_to_end_seconds=time.monotonic() - started, **evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-compute", help="run multiply/ReLU and check every result")
    verify.add_argument("--target", default="cpu", help="cpu, cuda:N, rocm:N or xpu:N")
    source = verify.add_mutually_exclusive_group(required=True)
    source.add_argument("--library", help="absolute, reviewed CPU reference library")
    source.add_argument("--bundled", action="store_true", help="use hash-checked platform wheel CPU library")
    verify.add_argument("--elements", type=int, default=257)
    args = parser.parse_args()
    if not re.fullmatch(r"cpu|(cuda|rocm|xpu):[0-9]{1,3}", args.target):
        parser.error("select cpu, cuda:N, rocm:N or xpu:N")
    if not 1 <= args.elements <= 65536:
        parser.error("elements must be 1..65536")
    try:
        result = verify_compute(args)
    except BackendUnavailable as error:
        print(json.dumps(dict(format="hyperl-hardware-check/1", status="unavailable",
                              target=args.target, reason=str(error))))
        return 3
    except (ValueError, RuntimeError, OSError, MemoryError) as error:
        print(f"HyperL verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
