# Reuse reviewed upstream components

The owner authorizes using existing web code/methods. Prefer maintained upstream
libraries and supported interfaces over copying implementations. Pin a release or
commit, preserve licenses/notices and source provenance, review transitive/native
components, and test the exact target. Public availability is not permission to
relicense closed or restricted source. No vendor performance transfers automatically.

The new HyperL licence does not replace dependency grants. Check compatibility
before combining newly restricted code with copyleft code; retain each component's
licence, notices and source obligations. See [scope history](LICENSE_HISTORY.md).

Already used: Kotlin standard library, kotlinx.serialization, kotlinx.coroutines,
JDK Swing/ZIP, JCE AES-GCM/SecureRandom/MessageDigest and JCA Ed25519, Gradle and
CMake. See the [cryptographic contracts](CRYPTOGRAPHY.md); availability of a
primitive does not imply key enrollment, replay protection or certification.
The workbench also links pinned FlatLaf 3.7.2 no-natives (Apache-2.0) and
RSyntaxTextArea 4.0.1 (BSD-3-Clause), with licenses under docs/licenses. Their
versions/notices are recorded. The small CPU reference is a conformance oracle;
it is not intended to replace optimized math/compiler libraries.

| Candidate | Intended reuse | Current state |
|---|---|---|
| [LLVM/MLIR](https://github.com/llvm/llvm-project/blob/main/LICENSE.TXT) | Compiler IR, CPU SIMD/code generation and GPU lowering; Apache-2.0 with LLVM exceptions | Research/next backend; not bundled |
| [Apache TVM](https://tvm.apache.org/), [license](https://github.com/apache/tvm/blob/main/LICENSE) | Tensor compiler/runtime and target-specific lowering instead of rewriting a complete compiler | Candidate integration, Apache-2.0; exact subcomponents/version need pinning |
| [ONNX Runtime execution providers](https://onnxruntime.ai/docs/execution-providers/), [MIT license](https://github.com/microsoft/onnxruntime/blob/main/LICENSE) | Existing model runtime/providers, Windows DirectML and platform SDK adapters | Candidate; no standalone HyperL ONNX backend yet; individual provider dependencies differ |
| PyTorch/ExecuTorch and Triton | Model export, mobile runtime and qualified GPU compilation | See RESEARCH_2026_10_08.md; licenses/targets are component-specific, not universal device support |
| CUDA/HIP, CUTLASS, Metal, Vulkan, OpenCL | Supported SDK interfaces and licensed kernel libraries | OpenCL/Metal bridges and native Android Vulkan runner implemented; narrow Radeon/Adreno results documented; CUDA/HIP remain source/optional framework paths; drivers/SDKs separate |
| UCX/hwloc/Open MPI or similar | Mature transports/topology/resource handling | Future authenticated host adapters, with shutdown/isolation/conformance and measured load |
| [free5GC](https://github.com/free5gc/free5gc/blob/main/LICENSE) | Independently installed 5G core research candidate, Apache-2.0 top-level license | Not installed; separately review network functions, dependencies, privileges and lab scope |
| OAI/srsRAN/Open5GS | Proven telecom lab stacks/interfaces | Separate licensed components/services; not copied into Apache tree or certified by HyperL |
| Odysseus | Optional operator workspace and design reference | AGPL project, separately reviewed in Meshlit; no source/assets bundled |

The 2026-10-24 [quantum prototype](QUANTUM_EXTENSION_PLAN.md) adds original bounded
stdlib state-vector simulation and thin optional IBM, Cirq/Google, Azure IonQ and
OriginQ SDK interfaces. It does not copy provider implementations or install their
SDKs automatically. Provider fixture tests are not SDK acceptance or physical QPU
evidence; selected SDK licenses, versions, transport and actual jobs need separate
qualification. OriginQ explicitly configures HTTPS rather than accepting its
[documented HTTP default](https://raw.githubusercontent.com/OriginQ/pyqpanda3-skill/main/resources/api/qcloud/service.md).

The next model/compiler milestone should start with a thin adapter to one reviewed
runtime, a real model/kernel fixture and observed hardware results. Keep runtime
operators in that upstream implementation; maintain HyperL's IR translation,
qualification, permissions, scheduling and evidence locally. Research candidates
are not installed dependencies or successful execution results.
