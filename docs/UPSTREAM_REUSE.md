# Reuse reviewed upstream components

The owner authorizes using existing web code/methods. Prefer maintained upstream
libraries and supported interfaces over copying implementations. Pin a release or
commit, preserve licenses/notices and source provenance, review transitive/native
components, and test the exact target. Public availability is not permission to
relicense closed or restricted source. No vendor performance transfers automatically.

Already used: Kotlin standard library, kotlinx.serialization, kotlinx.coroutines,
JDK Swing/ZIP and JCE AES-GCM/SecureRandom/MessageDigest, Gradle and CMake. Their
versions/notices are recorded. The small CPU reference is a conformance oracle;
it is not intended to replace optimized math/compiler libraries.

| Candidate | Intended reuse | Current state |
|---|---|---|
| [LLVM/MLIR](https://github.com/llvm/llvm-project/blob/main/LICENSE.TXT) | Compiler IR, CPU SIMD/code generation and GPU lowering; Apache-2.0 with LLVM exceptions | Research/next backend; not bundled |
| [Apache TVM](https://tvm.apache.org/), [license](https://github.com/apache/tvm/blob/main/LICENSE) | Tensor compiler/runtime and target-specific lowering instead of rewriting a complete compiler | Candidate integration, Apache-2.0; exact subcomponents/version need pinning |
| [ONNX Runtime execution providers](https://onnxruntime.ai/docs/execution-providers/), [MIT license](https://github.com/microsoft/onnxruntime/blob/main/LICENSE) | Existing model runtime/providers, Windows DirectML and platform SDK adapters | Candidate; no standalone HyperL ONNX backend yet; individual provider dependencies differ |
| PyTorch/ExecuTorch and Triton | Model export, mobile runtime and qualified GPU compilation | See RESEARCH_2026_10_08.md; licenses/targets are component-specific, not universal device support |
| CUDA/HIP, CUTLASS, Metal, Vulkan, OpenCL | Supported SDK interfaces and licensed kernel libraries | OpenCL host bridge implemented; others source emission/research only; drivers/SDKs separate |
| UCX/hwloc/Open MPI or similar | Mature transports/topology/resource handling | Future authenticated host adapters, with shutdown/isolation/conformance and measured load |
| [free5GC](https://github.com/free5gc/free5gc/blob/main/LICENSE) | Independently installed 5G core research candidate, Apache-2.0 top-level license | Not installed; separately review network functions, dependencies, privileges and lab scope |
| OAI/srsRAN/Open5GS | Proven telecom lab stacks/interfaces | Separate licensed components/services; not copied into Apache tree or certified by HyperL |
| Odysseus | Optional operator workspace and design reference | AGPL project, separately reviewed in Meshlit; no source/assets bundled |

The next model/compiler milestone should start with a thin adapter to one reviewed
runtime, a real model/kernel fixture and observed hardware results. Keep runtime
operators in that upstream implementation; maintain HyperL's IR translation,
qualification, permissions, scheduling and evidence locally. Research candidates
are not installed dependencies or successful execution results.
