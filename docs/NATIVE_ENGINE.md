# Native engine contracts

## Projection and admission

The CLI accepts a local immutable regular file, expected SHA-256, input and weight names, a tensor-byte budget, and either `auto`, an explicit number of CPU workers, or a GPU selector. Prevent writes and truncation for the entire lifetime of the mapping. Rust views borrow bytes directly from the mapping and accommodate unaligned tensor offsets. Header parsing is capped at 4 MiB. Full digest verification touches all file pages. Native byte views support inspection of upstream SafeTensors dtypes, while execution accepts the encodings listed in the README.

For input [B,K] and weights [N,K], output is little-endian F32 [B,N] under the name `output`. Packed INT4 weights are physically [N,K/2]. A packed row requires even K. A scale of 0.5 applied to byte 0xe1 yields 0.5 and −1. CPU converts elements as needed. No full widened weight matrix is allocated. CPU admission charges the mapped file and result payload. OS mappings, thread stacks, allocator overhead, compiler and driver memory require additional headroom.

The optional wgpu path uses one 8×8 tiled compute kernel. Each workgroup reads blocks of the two matrices into shared memory and handles incomplete tiles. GPU admission also charges the GPU input, weights, result, readback and uniform. Dispatch is capped at 2³⁰ multiply-accumulates and checked against device storage and grid limits. F32 GPU arithmetic can differ from CPU rounding. Reject nonfinite inputs or output and compare finite results to an independent f64 reference during hardware qualification. Ctrl-C cancels CPU work at checkpoints. A submitted GPU dispatch can complete before cancellation is observed. Outputs are staged in the destination directory and published with no overwrite.

## Compile a graph

Use an isolated environment with the pinned JAX packages from `requirements/jax-test.txt`. Install `requirements/mlir-test.txt` for the optional IREE compiler/runtime on a supported host.

```python
import jax
import numpy as np
from hyperl.ir import Graph, Type, DType

g = Graph()
q = g.add_input("q", Type(DType.F32, (2, 3, 4)))
v = g.add_input("v", Type(DType.F32, (2, 3, 2)))
g.set_outputs([g.attention(q, q, v, causal=True)])
module = g.lower(max_tensor_bytes=4096)
run = module.compile(jax.devices("cpu")[0])
out, = run(np.ones((2, 3, 4), np.float32), np.ones((2, 3, 2), np.float32))
assert out.shape == (2, 3, 2)
```

Set `PYTHONPATH=python` when using a source checkout. The module contains verified MLIR in the StableHLO dialect. Inputs have static shapes up to rank four. Matmul uses f32 accumulation with highest requested dot precision. Attention accepts per-head matrices with matching leading dimensions, normalizes over keys, and optionally applies a causal mask for equal-length sequences. The graph also supports GEMV with bias, whole-tensor RMSNorm and whole-tensor softmax. Graph admission accounts for declared logical values and the dense attention score matrix. Lowering and backend temporaries have separate memory costs.

`module.iree("llvm-cpu")` returns a compiled VM flatbuffer. Explicit target options also include `vulkan-spirv`, `metal-spirv` and `cuda`, subject to support in the installed IREE build. Target selection and compilation alone do not qualify a device. The direct Rust projection runs independently of either Python compiler.

## Archive format

HLZ1 starts with four ASCII magic bytes and the original size as a little-endian u64. The stream covers consecutive plaintext blocks of up to 65,536 bytes. Each has a u8 codec tag, a little-endian u32 payload length and its payload. Tag 0 stores bytes. Tags 1, 2 and 3 use Huffman, range coding and rANS from the pinned constriction library. For each coded block the payload starts with 256 little-endian u32 symbol counts followed by little-endian u32 coded words. See the integer normalization in the Rust implementation. The final 64 bytes contain SHA-256 of the preceding wire bytes followed by SHA-256 of the concatenated plaintext. These unkeyed digests detect corruption. Use an independently trusted model digest at load time.

Manual modes are `raw`, `huffman`, `arithmetic`, and `ans`. Auto tries all three coders against raw and accounts for their headers. Sampling time and the storage ratio vary with entropy. The final CLI byte limit applies to plaintext. Temporary disk space must fit the full new artifact or decoded file. Loading compressed files first requires verified unpacking.

## Performance qualification

Run `python benchmarks/run_benchmarks.py --device cpu` in the graph environment. For a CUDA comparison, use a qualified NVIDIA machine with both CUDA-enabled JAX and CUDA-enabled PyTorch and set `CUDA_VISIBLE_DEVICES` to a single GPU. Add `--device cuda --size 1024 --rounds 20`. Both paths run the same seeded f32 product, include host input and host output, synchronize and compare every sample against a float64 oracle. The report records compilation separately. It measures one operator and does not establish whole-model decode performance. The prior synthetic token benchmark derived competitor rates from HyperL timings and has been retired.
