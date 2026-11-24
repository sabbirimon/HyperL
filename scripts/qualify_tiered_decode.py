"""Qualify bounded Metal/RAM/file-backed layer and KV placement on the pinned SmolLM2 fixture."""

# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from hyperl.llm import file_digest
from hyperl.model_storage import MemoryTier, ModelLayer, Workload, plan_model

MODEL_SHA = "2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d"
PROMPT = [6403, 1980, 253, 655]
MIB = 1024**2


def placement(sizes, gpu_bytes, ram_bytes):
    tiers = [
        MemoryTier(
            "metal", "mac", "GPU", gpu_bytes, 1.0e9, "discrete-vram", "torch-mps"
        ),
        MemoryTier("ram", "mac", "RAM", ram_bytes, 1.0e8, "system-ram"),
        MemoryTier("ssd", "mac", "STORAGE", sum(sizes), 1.0e7, "scratch"),
    ]
    layers = [ModelLayer(str(i), size) for i, size in enumerate(sizes)]
    plan = plan_model(
        layers, tiers, Workload(reserve_fraction=0, allow_storage_offload=True)
    )
    return [p["storage_tier"] for p in plan["placements"]]


def qualify(source, folder, tokens):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from gguf import GGUFReader, dequantize
    from safetensors.torch import load_file, save_file

    if file_digest(source) != MODEL_SHA:
        raise ValueError(
            "This fixture requires the pinned SmolLM2 135M Instruct Q4_K_M model"
        )
    if platform.system() != "Darwin" or platform.machine() != "x86_64":
        raise RuntimeError(
            "This qualification targets an Intel Mac with discrete GPU memory"
        )
    if (
        not torch.backends.mps.is_available()
        or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
    ):
        raise RuntimeError(
            "An explicitly available MPS device with CPU fallback disabled is required"
        )
    torch.set_num_threads(2)
    reader = GGUFReader(str(source))
    tensors = {entry.name: entry for entry in reader.tensors}

    def unpack(name):
        t = tensors[name]
        return torch.from_numpy(
            np.ascontiguousarray(dequantize(t.data, t.tensor_type)).copy()
        ).float()

    # The fixed digest pins the shape, tied output embedding and interleaved GGUF RoPE layout.
    embed = unpack("token_embd.weight")
    norm = unpack("output_norm.weight")
    keys = [
        "attn_norm",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "ffn_norm",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
    ]
    sizes = [
        sum(int(np.prod(tensors[f"blk.{i}.{k}.weight"].shape)) * 4 for k in keys)
        for i in range(30)
    ]
    # A real per-process Metal allocator cap keeps pressure local to this experiment.
    torch.mps.set_per_process_memory_fraction(0.09)
    trial = []
    try:
        for size in sizes:
            trial.append(torch.empty(size, device="mps", dtype=torch.uint8))
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        full_resident_rejected_after = len(trial)
    else:
        raise RuntimeError(
            "All block weights fit the Metal cap on this host; pressure was not demonstrated"
        )
    finally:
        trial.clear()
        torch.mps.empty_cache()
    weight_tiers = placement(sizes, 128 * MIB, 96 * MIB)
    context = len(PROMPT) + tokens
    kv_size = 2 * context * 3 * 64 * 4
    cache_tiers = placement([kv_size] * 30, kv_size * 12, kv_size * 8)
    if set(weight_tiers) != {"metal", "ram", "ssd"} or set(cache_tiers) != set(
        weight_tiers
    ):
        raise RuntimeError("Expected all three storage tiers to be exercised")
    resident, caches, disk_paths = [], [], []
    for i, (weights, cache) in enumerate(zip(weight_tiers, cache_tiers)):
        block = {k: unpack(f"blk.{i}.{k}.weight") for k in keys}
        path = folder / f"block-{i}.safetensors"
        if weights == "ssd":
            save_file(block, str(path))
            resident.append(None)
        else:
            device = "mps" if weights == "metal" else "cpu"
            resident.append({k: v.to(device) for k, v in block.items()})
        disk_paths.append(path)
        shape = (2, context, 3, 64)
        if cache == "ssd":
            arr = np.memmap(folder / f"kv-{i}.bin", dtype="<f4", mode="w+", shape=shape)
            caches.append(torch.from_numpy(arr))
        else:
            caches.append(
                torch.empty(shape, device="mps" if cache == "metal" else "cpu")
            )
    del block
    positions = torch.arange(context)[:, None] * (
        100000.0 ** (-torch.arange(0, 64, 2) / 64)
    )
    cos, sin = positions.cos().to("mps"), positions.sin().to("mps")

    def rotate(x, pos):
        pairs = x.reshape(-1, 32, 2)
        a, b = pairs.unbind(-1)
        return torch.stack(
            [a * cos[pos] - b * sin[pos], a * sin[pos] + b * cos[pos]], -1
        ).flatten(1)

    def rms(x, scale):
        return x * torch.rsqrt(x.square().mean() + 1.0e-5) * scale

    def step(token, pos):
        x = embed[token].to("mps")
        for i, backing in enumerate(weight_tiers):
            stored = load_file(str(disk_paths[i])) if backing == "ssd" else resident[i]
            w = {k: v.to("mps") for k, v in stored.items()}
            h = rms(x, w["attn_norm"])
            q = rotate(w["attn_q"].mv(h), pos)
            k = rotate(w["attn_k"].mv(h), pos)
            v = w["attn_v"].mv(h).reshape(3, 64)
            cache = caches[i]
            cache[0, pos].copy_(k)
            cache[1, pos].copy_(v)
            past = cache[:, : pos + 1].to("mps").repeat_interleave(3, dim=2)
            scores = (past[0] * q).sum(-1) / 8
            context_vec = (scores.softmax(0).unsqueeze(-1) * past[1]).sum(0).flatten()
            x = x + w["attn_output"].mv(context_vec)
            h = rms(x, w["ffn_norm"])
            x = x + w["ffn_down"].mv(F.silu(w["ffn_gate"].mv(h)) * w["ffn_up"].mv(h))
            if backing != "metal":
                torch.mps.synchronize()
            del stored, w
        # The tied vocabulary projection is explicitly accounted as host work.
        return int(embed.mv(rms(x.cpu(), norm)).argmax())

    produced = []
    with torch.inference_mode():
        torch.mps.synchronize()
        start = time.perf_counter()
        for pos, token in enumerate(PROMPT):
            token_out = step(token, pos)
        first = time.perf_counter()
        produced.append(token_out)
        for pos in range(len(PROMPT), context - 1):
            produced.append(step(produced[-1], pos))
        elapsed = time.perf_counter() - first
    oracle_path = (
        Path(__file__).resolve().parents[1]
        / "docs/evidence/model-sharding-2026-11-02.json"
    )
    oracle = json.loads(oracle_path.read_text())
    if (
        oracle["model_sha256"] != MODEL_SHA
        or produced != oracle["local_tokens"][:tokens]
    ):
        raise RuntimeError(
            f"Tiered decoder differs from independently recorded native CPU tokens: {produced}"
        )
    return dict(
        model_sha256=MODEL_SHA,
        prompt_tokens=PROMPT,
        generated_tokens=produced,
        pytorch=torch.__version__,
        weight_layer_counts=dict(Counter(weight_tiers)),
        kv_layer_counts=dict(Counter(cache_tiers)),
        block_resident_budget_bytes={"metal": 128 * MIB, "ram": 96 * MIB},
        kv_resident_budget_bytes={"metal": 12 * kv_size, "ram": 8 * kv_size},
        metal_memory_fraction=0.09,
        all_resident_allocation_failed_after_blocks=full_resident_rejected_after,
        metal_driver_bytes_after_decode=torch.mps.driver_allocated_memory(),
        host_embedding_bytes=embed.numel() * 4,
        file_backed_weight_bytes=sum(
            p.stat().st_size for p in disk_paths if p.exists()
        ),
        file_backed_kv_bytes=cache_tiers.count("ssd") * kv_size,
        first_token_seconds=first - start,
        decode_tokens_per_second=(tokens - 1) / elapsed,
        scope="Application budgets on Intel Mac with discrete GPU. Warm model, OS file caching permitted.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--tokens", type=int, default=8, choices=range(2, 17))
    args = p.parse_args()
    with tempfile.TemporaryDirectory(prefix="hyperl-tiered-decode-") as scratch:
        print(
            json.dumps(
                qualify(args.model.resolve(), Path(scratch), args.tokens), indent=2
            )
        )
