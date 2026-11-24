"""Compare three prebuilt, local dependency-planning executables on Linux or macOS."""
import argparse
from array import array
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time

def reference(jobs):
    finish = array("i", [1])
    checksum = 1
    for job in range(1, jobs):
        parents = [job - 1 - (job * factor % period) % job
                   for factor, period in ((4051, 65521), (2654435761, 4093))]
        end = max(finish[p] for p in parents) + 1 + job % 13
        finish.append(end)
        checksum += end
    return checksum

def run(executable, expected):
    with tempfile.TemporaryFile() as out:
        start = time.perf_counter_ns()
        child = subprocess.Popen([str(executable)], stdout=out)
        _, status, usage = os.wait4(child.pid, 0)
        ms = (time.perf_counter_ns() - start) / 1e6
        child.returncode = os.waitstatus_to_exitcode(status)
        out.seek(0)
        if child.returncode or out.read(100) != f"{expected}\n".encode():
            raise ValueError(f"wrong scheduling result from {executable.name}")
        rss = usage.ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
        return ms, rss

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binaries", type=Path, help="directory containing goose, cpp and rust")
    args = parser.parse_args()
    if platform.system() not in {"Darwin", "Linux"}:
        raise SystemExit("per-process peak RSS requires Linux or macOS")
    jobs = 4_000_000
    checksum = reference(jobs)
    names = ["goose", "cpp", "rust"]
    results = {name: [] for name in names}
    for i in range(10):
        for name in names[i % 3:] + names[:i % 3]:
            row = run(args.binaries.resolve() / name, checksum)
            if i:
                results[name].append(row)
    report = {"jobs": jobs, "checksum": checksum, "platform": platform.system(), "machine": platform.machine(),
              "scope": "single-process offline DAG earliest-finish pass including startup and teardown",
              "measurements": {}}
    for name, pairs in results.items():
        times, memory = zip(*pairs)
        report["measurements"][name] = {"samples_ms": times, "median_ms": statistics.median(times),
                                        "median_peak_rss_bytes": statistics.median(memory)}
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
