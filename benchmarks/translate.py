# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
"""Compare end-to-end CUDA migration using two translator checkouts, with artifact parity."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def signature(folder):
    report = json.loads((folder / "migration.json").read_text())
    files = {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in folder.rglob("*") if p.is_file() and p.name != "migration.json"}
    return files, [(row["path"], row["sha256"], row.get("upstreamWarnings") or 0) for row in report["sources"]], report.get("interface")


def compare(roots, engine, repeats):
    cohorts = [(1, "hip"), (24, "hip"), (128, "hip"), (1, "metal")]
    result = []
    with tempfile.TemporaryDirectory() as tmp:
        for count, target in cohorts:
            work = Path(tmp) / f"{target}-{count}"
            source = work / "input"
            source.mkdir(parents=True)
            for i in range(count):
                code = (ROOT / "examples/cuda/add.cu").read_text().replace("void add(", f"void sum{i}(")
                if target == "hip":
                    code = "#include <cuda_runtime.h>\n" + code
                    code += f"cudaError_t synchronize{i}() {{ return cudaDeviceSynchronize(); }}\n"
                    if i % 8 == 7:
                        code += "void unresolved() { cudaUnknownForBenchmark(); }\n"
                (source / f"kernel{i}.cu").write_text(code)
            samples = [[], []]
            expected = None
            for iteration in range(repeats + 1):
                for idx in (range(2) if iteration % 2 == 0 else reversed(range(2))):
                    dest = work / f"{iteration}-{idx}"
                    inp = source if count > 1 else source / "kernel0.cu"
                    start = time.perf_counter_ns()
                    subprocess.run([sys.executable, roots[idx] / "studio/cuda_migrate.py", inp, dest,
                                    "--engine", engine, "--target", target], check=True, stdout=subprocess.DEVNULL, timeout=200)
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    current = signature(dest)
                    if expected is None:
                        expected = current
                    if current != expected:
                        raise RuntimeError(f"Artifact or diagnostics mismatch on {count}-file {target}")
                    if iteration:
                        samples[idx].append(round(elapsed, 2))
            a, b = map(statistics.median, samples)
            result.append({"target": target, "files": count, "sourceBytes": sum(p.stat().st_size for p in source.iterdir()),
                           "baselineMs": samples[0], "candidateMs": samples[1], "medianBaselineMs": a,
                           "medianCandidateMs": b, "speedup": round(a / b, 2)})
    return {"host": platform.platform(), "python": platform.python_version(), "warmups": 1, "repeats": repeats, "results": result}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--repeats", type=int, choices=range(3, 32), default=5)
    args = p.parse_args()
    print(json.dumps(compare([args.baseline.resolve(), ROOT], args.engine.resolve(), args.repeats), indent=2))
