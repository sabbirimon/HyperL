# Goose scheduler experiment

[CUDA migration and accelerator support](../../docs/CUDA_MIGRATION.md)


This benchmark computes earliest finish times for four million tasks in topological order. Each task depends on two prior tasks with deterministic indices and has a known duration. All three executables use one flat array of 32-bit finish times and accumulate a 64-bit checksum. Rust and C++ reserve their full capacity up front. The harness compares every result with an independent Python implementation, rotates execution order, discards one warmup per language and measures nine fresh processes per language. Peak RSS is physical memory, distinct from Goose's reserved virtual address space.

Build the upstream Goose compiler at revision `2239b92eb2ae0e8f390f98640b09ff57a17698fc` in Release mode. The experiment needs no optional Goose packages. From the HyperL checkout, using an independently installed compiler:

```sh
mkdir -p build/goose-trial
goose --standalone -o build/goose-trial/dependencies.c benchmarks/goose/dependencies.goose
clang -O3 build/goose-trial/dependencies.c -lm -pthread -o build/goose-trial/goose
clang++ -O3 -std=c++20 benchmarks/goose/dependencies.cpp -o build/goose-trial/cpp
rustc -C opt-level=3 benchmarks/goose/dependencies.rs -o build/goose-trial/rust
python3 benchmarks/goose/compare.py build/goose-trial
```

| Local median | Time | Peak RSS |
| --- | ---: | ---: |
| C++ | 71.72 ms | 15.84 MiB |
| Goose | 88.97 ms | 15.81 MiB |
| Rust | 105.08 ms | 15.92 MiB |

All produced checksum 38,134,856,341. On this Intel Mac, Goose improved on the Rust implementation and C++ finished first. The near-equal RSS is consistent with all three using the same compact representation. Keep Goose as an optional experiment. A stronger next case is variable-sized dependency and lease records with real traces, where its inline storage might matter. This synthetic dependency pass excludes network traffic, model execution, concurrent lease updates and GPU kernels. [Recorded samples and versions](../../docs/evidence/goose-scheduler-2026-11-23.json) · [Upstream benchmark suite](https://github.com/aardappel/goose/blob/2239b92eb2ae0e8f390f98640b09ff57a17698fc/bench/results.md)
