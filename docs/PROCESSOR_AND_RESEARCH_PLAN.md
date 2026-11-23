# Processor management and research intake

[CUDA migration and accelerator support](CUDA_MIGRATION.md)

The bounded SmolLM2 pressure test exercised automatic weight and KV placement on three tiers. Integrating the qualified approach into general model loading and measuring other hosts remain open. [Three-tier decode](LOGICAL_DEVICES_AND_MODEL_STORAGE.md#three-tier-decode-on-intel-macos).


## Architecture decision

Give Rust ownership of admission, buffer lifetimes, cancellation, and publication. Keep a narrow C ABI for existing native kernels. Use standard StableHLO operations as the graph interchange so a new backend can consume typed graphs without its own source emitter. Keep a separately testable direct projection for devices running without a compiler. An optional wgpu backend can reuse one tiled kernel across operating systems. Hardware support requires a completed dispatch and a checked result.

## Processor management

| Stage | Decision and acceptance criterion |
| --- | --- |
| Local CPU | Read the OS parallelism limit. Divide independent output cells among at most 64 workers, including for a single-token projection. Honor explicit worker limits and reject requests above capacity. Charge a mapped artifact once and reserve the output before starting. Repeat one and many worker executions against the same answer. |
| GPU | Enumerate hardware with backend and device identity. Admit buffer sizes, dispatch dimensions, upload and readback against the budget. Require an exact adapter selection to reproduce a run. Test a partial workgroup and a non-square projection on each physical backend. |
| Topology and power | Add observed memory domains, cache and NUMA locality, CPU affinity restrictions and sustained thermal throttling as optional capabilities. Benchmark cold load, steady-state throughput and tail latency with shared RAM counted once. Keep a conservative default on systems with missing telemetry. |
| Workload policy | Fit weights, KV cache, activations, output and transfer staging before ranking candidates. Decode favors resident weights and short queues. Prefill favors dense throughput. Offline jobs can batch requests or stage checked cold weights. Record the observations and allow a manual backend, device, threads, memory cap and latency target. |
| Fleet | Feed measured local capacity into existing cell leases and versioned model profiles. Reserve capacity at the serving node, release on completion, fence a lost device and reject stale results. Qualify interrupted and concurrent jobs on two physical hosts before expanding cells. A million directory rows exercises storage only. |

## Research decisions

| Candidate | Decision and evidence to collect |
| --- | --- |
| Entropy coding | Adopt versioned, independently coded blocks for cold storage or transport. Compare the actual encoded sizes with raw storage. Require bounded decode, cancellation, corruption detection and identical restored artifact hashes. Hot mapped artifacts retain their original container. |
| OpenAI GPU kernel optimization | Apply tiling and differential numerical checks to the projection. Measure real hardware, include transfers and use an actual vendor implementation for comparisons. Triton and FpSan are candidates for a future NVIDIA-specific experiment. [OpenAI engineering report](https://openai.com/index/gpt-5-6-frontier-intelligence-efficiency/) |
| Claude scientific computing and compiler work | Adopt an independent numerical oracle, run the same input through each backend, and retain small reproducing fixtures. Those practices work with an offline build and allow the compiler to change. [Scientific computing study](https://www.anthropic.com/research/long-running-claude) · [Compiler project](https://www.anthropic.com/research/building-c-compiler) |
| Matrix multiplication, family 107 | Compare arithmetic model, dimensions, precision, workspace and crossover before considering a kernel. Real matrix-product workloads here include narrow, rectangular decode projections. |
| Integer multiplication, family 109 | Revisit for a measured multiprecision bottleneck. The preprint gives an asymptotic bound for a fixed multitape machine. Current dot products use machine words and floating point. |
| Online k-server, family 110 | Candidate for trace replay of paging decisions. Model transfer costs as a metric first. Compare transferred bytes, misses, hot-weight evictions and p95 latency to the existing cache. |
| Scheduling, family 124 | Candidate for an offline reference for three identical processors. Establish which job and precedence assumptions a trace satisfies before comparing it to heterogeneous online placement. |
| ETH, family 378 | Track the proof and assumptions for exact combinatorial planning research. A dispatch policy still needs a bounded solver, a deadline, a fallback, and measured quality. [OpenAI research release](https://openai.com/index/building-scientific-understanding-with-academic-communities/) |
| OpenAI/AMD RTL | Revisit only for an identified hardware block, licensed RTL, simulation oracle and synthesis timing, power and area constraints. AMD's reported 14% concerns specific IP. [Hardware study](https://openai.com/index/advancing-chip-design-with-our-models/) |

The [offline index](research/manuscripts.csv) contains one pinned PDF link per manuscript. It covers 721 manuscripts in 373 families. The index records publication metadata. Individual proof validity, practical crossover, rights to redistribute code and deployment readiness require separate review. To revisit a family, filter the CSV by family ID and read the linked manuscript and corrections. Rebuild the index with `python3 scripts/index_math_catalog.py overview.tex index.csv` using the pinned [source catalog](https://github.com/openai/math/blob/4eaee87548250b2682affda7d05d947c05554bae/overview.tex).

## Model-proposed weight edits

Fan et al. tested Qwen3.5 at 2B, 4B, 9B and 27B. On their held-out evaluation the strongest 27B edit gained 6.6 percentage points. It changed a bias specific to the model's linear-attention layers. The smaller models either declined to edit or degraded. The authors kept the editor frozen while it proposed and compared edits to another instance. [Paper, methods and evaluation](https://arxiv.org/html/2611.21885v2)

HyperL can create a separate candidate by clipping one named floating-point tensor and can run it through a supported local adapter. Model-directed repair still requires architecture-specific probes, access to hidden activations and a way for the model to propose and evaluate its edits. Start with a local floating-point checkpoint and store candidate changes separately, keyed to the original digest. Compare scores with matched prompts, seeds and token budgets. Reserve separate held-out examples, check ordinary chat and unrelated tasks for regressions, and report evaluation cost. Promote an edit only after it passes these checks. A GGUF export needs its own post-quantization evaluation before it can be served in Meshlit.

Meshlit can present evaluation results for any compatible text endpoint. Editing the served weights additionally requires control of the checkpoint and inference backend. An API-only provider can propose an edit to a different local checkpoint. That would be cross-model editing. Neither path has been qualified in this project.

### Checkpoint repair versus behavioral recovery

The authors report that on six image benchmarks a damaged copy recovered from about 7% to 70% accuracy, roughly 83% of the intact model's accuracy. This measures recovered behavior. The trusted original bytes were never read by the editor. The experiment begins with a trained checkpoint and a runnable model that can diagnose a damaged copy of itself. A missing checkpoint cannot be regenerated from its manifest or tokenizer. [Authors' experiment](https://www.jiajunfan.com/projects/self-rewriting/)

HyperL already protects checkpoint bytes through a trusted protection manifest and parity fragments. On 2026-11-24 a test corrupted a stored fragment of a tiny Flax GPT-2 checkpoint, repaired it, exported identical SHA-256 bytes and decoded the original four tokens through the public Flax adapter. This qualifies storage repair only.

On a separate tiny SafeTensors GPT-2 fixture, one deliberately damaged embedding value suppressed all visible output for a known prompt. A human selected a clipping limit of 2.0. HyperL changed one value in a separate candidate, and the public adapter generated the original four tokens again. This trial used one prompt and supplies no held-out accuracy result. See the [candidate command](USER_GUIDE.md#experimental-checkpoint-edit).

For a model-directed behavioral-recovery trial, start with a writable local floating-point SafeTensors model, a deliberately damaged disposable copy and disjoint practice and held-out examples. Compare intact, damaged and edited variants under identical decoding settings. Record the damaged tensor, proposed numerical edit, original and candidate digests, evaluation cost, and scores on target and unrelated tasks. Accept only bounded tensor operations in the experiment. Keep the original available for rollback. A claim of self-rewriting requires the same model family and checkpoint to propose its own edit and an independent evaluation of that edit. The 27B result is still unreplicated here.

## Goose language trial

The [matched scheduling experiment](../benchmarks/goose/README.md) found C++ fastest on the flat dependency table, followed by Goose and Rust. Goose compiles to C and can call native C functions, making an isolated metadata pass a viable future experiment. Keep the shipped Rust engine and C ABI while evaluating variable-size record workloads. [Goose compiler and native interface](https://github.com/aardappel/goose/tree/2239b92eb2ae0e8f390f98640b09ff57a17698fc)
