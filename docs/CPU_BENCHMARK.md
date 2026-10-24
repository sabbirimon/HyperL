# Local CPU baseline — 2026-10-08

A real Release C build on macOS 15.8.1/x86-64, AppleClang 17, Python 3.12.4
and NumPy 2.3.3 executes finite f32 multiply then ReLU. The CPU brand query is
unavailable under this session's permissions and is recorded as unknown. Each
path's output matches the oracle exactly before measurement. Three warmups and
nine batches of twenty calls produce the medians below, in **microseconds**.

| Elements | Python API | Native ABI | NumPy pipeline | Preallocated naive C kernel |
|---:|---:|---:|---:|---:|
| 128 | 44.149 | 3.979 | 1.731 | 1.204 |
| 4096 | 607.140 | 44.978 | 12.897 | 2.030 |
| 65536 | 10132.175 | 875.063 | 311.647 | 18.528 |

These columns have different cost scopes. Python includes validation, private input
snapshots, ctypes setup and native allocation/computation/output. Native ABI excludes
caller preparation but includes C validation/allocation/copying. NumPy includes two
ufuncs and intermediate/output allocation. The naive C baseline excludes validation,
admission and allocation, with one preallocated loop. All timing includes the relevant
Python/ctypes call overhead. It is not appropriate to describe their ratios as
identical complete workloads or a production application speedup.

The admitted HyperL path is slower than NumPy in this observation. Removing repeated
setup/copies and defining reusable allocation lifetimes are concrete optimization
work; the baseline does not conceal this overhead. Paths run sequentially, the host
is not isolated, and other validation processes may be active. No CPU affinity,
thermal or sustained-load qualification is claimed. The raw p95 is across **batch
means**, not individual-call tail latency. This is neither a GPU comparison nor
LLM/training, distributed, dataset-IO or end-to-end application throughput.

The separate accuracy example `[16777216, 1, -16777216]` returns 0 with the unchanged
ordered f32 graph sum and 1 with compensated `precise-sum/1`, matching `math.fsum`
for that example. This is an accuracy choice rather than a general performance claim.

[Raw samples, exact binary hashes and scopes](evidence/cpu-benchmark-2026-10-08.json).
The source-controlled report contains synthetic data and no private paths or keys.
Run a new report on the selected actual host; do not overwrite this observation:

```sh
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release -DHYPERL_BENCHMARK_REFERENCE=ON
cmake --build build/native --config Release
python scripts/benchmark_cpu.py \
  --library /absolute/path/to/libhyperl_cpu_runtime.so \
  --reference /absolute/path/to/libhyperl_benchmark_reference.so \
  --output /absolute/new-cpu-report.json
```

Use the actual `.dylib` or `.dll` names on macOS/Windows. NumPy 2.3.3 is the pinned
comparison dependency; no library, framework or compiler is downloaded by this
script. The opt-in benchmark baseline is not installed as a production API.
