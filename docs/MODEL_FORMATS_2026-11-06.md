# Model formats, adapters and future compatibility

Owner request, 2026-11-06: support broader and future model formats as part of
HyperL's model engine. Meshlit Studio consumes this capability through the planned
versioned client interface. This does not change existing mobile defaults.

## Format is not an inference engine

`llama.cpp` is an engine; GGUF is a model container. SafeTensors stores tensors;
the architecture, tokenizer, generation configuration and runtime are additional
requirements. Sharded weight files are storage partitions, not evidence of
distributed execution. HyperL must retain these distinctions in its API and UI.

Keep independent fields for container/version, architecture, dtype/quantization,
tokenizer, model identity, available backend, hardware qualification and supported
sharding methods. A successful file import or header check cannot set execution
readiness. A device label or filename cannot establish backend compatibility.

## Implementation sequence

1. Add read-only `model-inspect` to HyperL's Python CLI and a versioned format
   registry. Report the depth of validation and missing runtime separately.
2. Add a bounded SafeTensors-to-existing-HyperL-JAX checkpoint import path for a
   deliberately narrow supported Llama architecture. Verify shapes, tensor names,
   dtype conversion, configuration and output digests before atomic publication.
3. Expose those observations through Studio's future HyperL connection, keeping
   its current local GGUF path functional. Add explicit import/convert actions;
   no automatic conversion or inference just from selecting a file.
4. Add independently qualified ONNX, LiteRT-LM and platform-native adapters in
   useful device order. Preserve models in original form; conversion is optional
   and requires a supported architecture, target and output validation.

## Format and backend matrix

| Family | First increment / planned treatment | Execution boundary |
| --- | --- | --- |
| GGUF | Existing native adapter; bounded header/version inspection | Only supported native architecture/quantization/backend profiles; header check is not full validation |
| SafeTensors, including index-sharded sets | Validate bounded tensor metadata, payload ranges and confined shard references; narrow Llama import | Conversion into the existing f32 JAX decoder is a separate operation; not arbitrary Hugging Face model support |
| HyperL JAX checkpoint | Existing manifest + f32 tensor format and tensor-parallel decoder | Explicit model digest, device mesh and memory budget; optional JAX runtime required |
| ONNX / external tensor data | Recognize candidate; dedicated parser/runtime adapter planned | Operators, external data, generation/KV protocol and execution provider need qualification |
| LiteRT / TFLite / LiteRT-LM | TFLite candidate recognition; LM package adapter planned | Tensor interpreter and conversational LM runtime have different contracts |
| Core ML `.mlmodel` / `.mlpackage` | Recognize candidate; Apple adapter planned | OS, model specification and device constraints must be checked |
| PyTorch `.pt` / `.pth` / ambiguous `.bin` | Recognize candidate only; no pickle deserialization | Explicit reviewed export or safe loading adapter required; never execute model-supplied Python |
| MLX model packages | Architecture/config/weights adapter planned | Often uses SafeTensors; filename recognition alone cannot establish MLX compatibility |
| OpenVINO IR / TensorRT engine packages | Dedicated adapters planned | Compiled targets, device/runtime versions and graph support are mandatory constraints |
| MindIR / vendor deployment artifacts | Separately qualified adapter planned | Vendor SDK and actual target hardware required; no generic phone/NPU promise |
| Future/unknown containers | Opaque checked storage remains usable; report unsupported parser/runtime | Explicit reviewed schema/parser and backend updates; no speculative loading |

The matrix is a staged plan, not a claim that all listed formats run now. Platform
adapters may support inference without supporting layer/tensor/expert partitioning.
Report those capabilities separately. Closed API services remain remote task
targets rather than locally loadable formats or shardable hidden weights.

## Versioned extension rules

- The inspector returns a versioned report with format identity, validation level,
  candidate adapters and execution/sharding limitations. Registry metadata alone
  never grants execution. Unknown container versions remain explicitly unknown.
- New format support is a reviewed parser plus conformance fixtures; new execution
  support is a separately reviewed adapter. Do not import plug-in code, fetch URLs,
  deserialize pickle or evaluate configuration while inspecting a model.
- Require bounded metadata, finite integer dimensions, consistent sizes, supported
  dtypes, duplicate-key rejection and confined shard/external-data paths. Reject
  invalid overlaps, traversal, symlinks and model mutation during validation where
  the operation claims immutable input. Preserve source and conversion provenance.
- Conversion must be explicit, bounded and cancellable, and must leave source files
  untouched. Reject unsupported architecture features instead of dropping tensors
  or changing their mathematical meaning. Verify output before publishing it.
- Future quantization schemes need their own dtype/operator support. A familiar
  container does not mean an unfamiliar packed tensor can execute correctly.
- Tests must separate metadata fixtures, converted synthetic-decoder equivalence,
  actual pretrained model generation, and hardware/sharding acceptance.

## Primary references reviewed

- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
- [SafeTensors documentation](https://huggingface.co/docs/safetensors/index) and
  [format implementation](https://github.com/safetensors/safetensors).
- [ONNX Runtime generation API](https://onnxruntime.ai/docs/genai/).
- [LiteRT and LiteRT-LM entry points](https://developers.google.com/edge/litert/overview).
- [Core ML conversion guide](https://apple.github.io/coremltools/docs-guides/source/introductory-quickstart.html).

These describe upstream capabilities, not HyperL qualification. Pin dependencies
and preserve their licences before adding an execution adapter.

## Delivered source/CLI checkpoint

The read-only inspector and narrow SafeTensors importer are now implemented.
Run from the standalone HyperL checkout or use its installed `python/` directory
as `PYTHONPATH`:

```sh
PYTHONPATH=python python3 -m hyperl.cluster model-inspect /absolute/model.gguf
PYTHONPATH=python python3 -m hyperl.cluster model-inspect /absolute/model.safetensors.index.json
PYTHONPATH=python /owner/jax-venv/bin/python -m hyperl.safetensors_import \
  /absolute/hf-model /absolute/new-hyperl-checkpoint --max-bytes 1073741824 --context 512
```

Inspection uses the standard library. Conversion requires NumPy; actual JAX
execution requires the separately provisioned environment in
`requirements/jax-test.txt`. No runtime dependencies are installed automatically.
The conversion CLI returns the manifest digest for the existing `JaxDecoder.load`
and model-profile APIs. Keep that digest as the output identity.

The converter accepts standard `LlamaForCausalLM`, dense bias-free SiLU/GQA,
untied embeddings and unscaled split-half RoPE. F16/BF16 values are explicitly
widened to f32; source files stay unchanged. Quantization, MoE, scaled/custom RoPE,
tied embeddings, unsupported tensor names/configuration and custom Python code
reject. It accepts one `model.safetensors` or a confined index of at most 128
shards. Metadata, source/output sizes and transpose working blocks are bounded;
cancellation/failure removes private staging. Successful output carries source
hashes and conversion provenance and passes the inspector's manifest contract.

This importer does not import a tokenizer: callers must supply matching token IDs.
It does not make arbitrary SafeTensors models runnable in Studio. Widening to f32
can significantly increase disk and memory requirements; the decoder still applies
its own memory budget. Conversion does not grant model or data licences.

Validation on 2026-11-06: **106 Python tests passed with zero skips** in the pinned
JAX environment, including eight format tests, nine importer tests and six existing
JAX tests. Converted F32/F16/BF16 synthetic Llama logits and greedy token IDs match
an independent reference; the existing tensor-parallel tests use four logical CPU
devices on this one Mac. No new pretrained SafeTensors model, GPU/TPU execution or
cross-device format conversion was qualified. Native CMake/CTest passed 5/5;
Gradle test/distribution build succeeded (unchanged JVM tests were up to date).

The installed Qwen3 GGUF inspector observed version 3, 398 tensors and 28 metadata
entries. It correctly reports header-only validation and does not claim execution
or sharding readiness. ONNX/TFLite/Core ML/PyTorch entries remain unverified
candidates requiring adapters; ambiguous `.bin` and unknown files remain unknown.
