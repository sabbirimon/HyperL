# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
"""Measure two release CLIs end to end, including verification and writing output."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import struct
import subprocess
import tempfile
import time


def fixture():
    tensors = {}
    data = bytearray()
    for name, rows, period, offset in [("x", 64, 7, 3), ("w", 256, 11, 5)]:
        start = len(data)
        data.extend(b"".join(struct.pack("<f", (i % period) - offset) for i in range(rows * 512)))
        tensors[name] = dict(dtype="F32", shape=[rows, 512], data_offsets=[start, len(data)])
    header = json.dumps(tensors).encode()
    header += b" " * (-len(header) % 8)
    return struct.pack("<Q", len(header)) + header + data


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("baseline", type=Path)
    cli.add_argument("candidate", type=Path)
    args = cli.parse_args()
    engines = [args.baseline.resolve(), args.candidate.resolve()]
    times = [[], []]
    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp) / "model.safetensors"
        raw = fixture()
        model.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        for pair in range(12):
            for index in ([0, 1] if pair % 2 == 0 else [1, 0]):
                output = Path(tmp) / "out.safetensors"
                start = time.perf_counter()
                subprocess.run([str(engines[index]), "project", str(model), digest, "x", "w",
                                "2000000", "1", str(output)], stdout=subprocess.DEVNULL, check=True)
                elapsed = (time.perf_counter() - start) * 1000
                result = hashlib.sha256(output.read_bytes()).hexdigest()
                if result != "3734e383994d5ad4b44bee497aa9489dbdb11949ab2479374b21b39a7f6231cb":
                    raise ValueError(f"Projection output differs: {result}")
                if pair:
                    times[index].append(elapsed)
                output.unlink()
    for engine, values in zip(engines, times):
        print(f"{engine}  {statistics.median(values):.2f} ms median ({len(values)} runs)")


if __name__ == "__main__":
    main()
