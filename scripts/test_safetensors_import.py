# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from hyperl.safetensors_import import convert, decoder_config

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_JAX = importlib.util.find_spec("jax") is not None


@unittest.skipUnless(HAS_NUMPY, "explicit optional NumPy environment required")
class SafetensorsImportTest(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source"; self.source.mkdir()
        self.target = self.root / "converted"
        self.cfg = dict(model_type="llama", architectures=["LlamaForCausalLM"], vocab_size=32, hidden_size=16,
                        intermediate_size=24, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=16, rms_norm_eps=1e-5, rope_theta=10000.0, hidden_act="silu",
                        attention_bias=False, mlp_bias=False, tie_word_embeddings=False)
        rng = np.random.default_rng(9)
        def matrix(shape):
            return rng.normal(0, .11, shape).astype(np.float32)
        self.weights = {"model.embed_tokens.weight": matrix((32, 16)), "model.norm.weight": np.ones(16, np.float32),
                        "lm_head.weight": matrix((32, 16))}
        for layer in range(2):
            p = f"model.layers.{layer}."
            self.weights[p + "input_layernorm.weight"] = np.ones(16, np.float32)
            self.weights[p + "post_attention_layernorm.weight"] = np.ones(16, np.float32)
            for key, shape in {"q": (16, 16), "k": (8, 16), "v": (8, 16), "o": (16, 16)}.items():
                self.weights[p + "self_attn." + key + "_proj.weight"] = matrix(shape)
            for key, shape in {"gate": (24, 16), "up": (24, 16), "down": (16, 24)}.items():
                self.weights[p + "mlp." + key + "_proj.weight"] = matrix(shape)
        self.write_config()

    def write_config(self):
        (self.source / "config.json").write_text(json.dumps(self.cfg))

    def write_shard(self, name, weights, dtype="F32"):
        np = self.np; header = {}; payload = bytearray(); rounded = {}
        for key, value in weights.items():
            if dtype == "BF16":
                raw = (value.view(np.uint32) >> 16).astype("<u2")
                rounded[key] = (raw.astype(np.uint32) << 16).view(np.float32)
            else:
                raw = value.astype("<f4" if dtype == "F32" else "<f2")
                rounded[key] = raw.astype(np.float32)
            start = len(payload); payload.extend(raw.tobytes())
            header[key] = dict(dtype=dtype, shape=list(raw.shape), data_offsets=[start, len(payload)])
        encoded = json.dumps(header).encode()
        encoded += b" " * (-len(encoded) % 8)
        (self.source / name).write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
        return rounded

    def run_import(self, **kwargs):
        return convert(self.source, self.target, max_bytes=1024 * 1024, chunk_bytes=1024, **kwargs)

    def assert_no_output(self):
        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.root.glob(".hyperl-import-*")), [])

    def test_all_precisions_real_values_and_source_provenance(self):
        for dtype in ("F32", "F16", "BF16"):
            with self.subTest(dtype=dtype):
                self.target = self.root / dtype
                rounded = self.write_shard("model.safetensors", self.weights, dtype)
                result = self.run_import(context=8)
                raw = (self.target / "manifest.json").read_bytes(); manifest = json.loads(raw)
                self.assertEqual(result["manifest_sha256"], hashlib.sha256(raw).hexdigest())
                self.assertEqual(manifest["config"]["context"], 8)
                self.assertEqual(manifest["provenance"]["original_context"], 16)
                self.assertEqual(manifest["provenance"]["source_dtypes"], [dtype])
                self.assertFalse(result["tokenizer_imported"])
                from hyperl.model_formats import inspect_model
                inspected = inspect_model(self.target)
                self.assertEqual(inspected["validation"], "manifest-validated-only")
                self.assertFalse(inspected["execution_ready"])
                self.assertEqual(manifest["provenance"]["sources"]["model.safetensors"], hashlib.sha256((self.source / "model.safetensors").read_bytes()).hexdigest())
                # Assert asymmetric source matrices directly, independent of converter's mapping helper.
                for dest, source in [("0.k", "model.layers.0.self_attn.k_proj.weight"),
                                     ("1.down", "model.layers.1.mlp.down_proj.weight"), ("head", "lm_head.weight")]:
                    actual = self.np.load(self.target / (dest + ".npy"), allow_pickle=False)
                    self.np.testing.assert_array_equal(actual, rounded[source].T)
                    self.assertEqual(actual.dtype, self.np.float32)
                    self.assertTrue(actual.flags.c_contiguous)

    def test_shards_index_and_path_confinement(self):
        names = list(self.weights)
        self.write_shard("a.safetensors", {k: self.weights[k] for k in names[:10]})
        self.write_shard("b.safetensors", {k: self.weights[k] for k in names[10:]})
        mapping = {k: ("a.safetensors" if k in names[:10] else "b.safetensors") for k in names}
        index = self.source / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": mapping}))
        self.run_import()
        self.target = self.root / "reject"
        mapping[names[0]] = "../a.safetensors"; index.write_text(json.dumps({"weight_map": mapping}))
        with self.assertRaises(ValueError): self.run_import()
        self.assert_no_output()

    def test_architecture_extensions_reject_before_output(self):
        self.write_shard("model.safetensors", self.weights)
        for key, value in [("rope_scaling", {"type": "linear", "factor": 2}), ("attention_bias", True),
                           ("mlp_bias", True), ("tie_word_embeddings", True), ("head_dim", 8), ("hidden_act", "gelu"),
                           ("model_type", "qwen2"), ("auto_map", {}), ("quantization_config", {}), ("pretraining_tp", 2)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                decoder_config({**self.cfg, key: value})
        with self.assertRaises(ValueError): self.run_import(context=32)
        self.assert_no_output()

    def test_missing_extra_wrongshape_nonfinite_and_symlink_reject(self):
        for variant in ("missing", "extra", "shape", "nan", "symlink"):
            with self.subTest(variant=variant):
                weights = dict(self.weights)
                if variant == "missing": weights.pop("lm_head.weight")
                if variant == "extra": weights["surprise"] = self.np.ones(3, self.np.float32)
                if variant == "shape": weights["lm_head.weight"] = self.np.zeros((16, 32), self.np.float32)
                if variant == "nan": weights["lm_head.weight"] = self.np.full((32, 16), self.np.nan, self.np.float32)
                path = self.source / "model.safetensors"
                if path.is_symlink(): path.unlink()
                self.write_shard(path.name, weights)
                if variant == "symlink":
                    path.rename(self.source / "other.safetensors"); path.symlink_to(self.source / "other.safetensors")
                with self.assertRaises(ValueError): self.run_import()
                self.assert_no_output()

    def test_overlap_trailing_duplicate_and_oversized_header_reject(self):
        path = self.source / "model.safetensors"
        for payload in [struct.pack("<Q", 2**32), struct.pack("<Q", 13) + b'{"a":1,"a":2}',
                        struct.pack("<Q", 2) + b"{}garbage"]:
            path.write_bytes(payload)
            with self.assertRaises(ValueError): self.run_import()
            self.assert_no_output()
        self.write_shard(path.name, self.weights)
        raw = path.read_bytes(); length = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8 + length]); header["model.norm.weight"]["data_offsets"] = [0, 64]
        encoded = json.dumps(header).encode(); path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8 + length:])
        with self.assertRaises(ValueError): self.run_import()
        self.assert_no_output()

    def test_quota_cancel_and_existing_destination(self):
        self.write_shard("model.safetensors", self.weights)
        with self.assertRaises(ValueError): convert(self.source, self.target, max_bytes=1024)
        calls = 0
        def cancel_midway():
            nonlocal calls
            calls += 1
            return calls > 12
        with self.assertRaises(InterruptedError): self.run_import(cancelled=cancel_midway)
        self.assert_no_output()
        self.target.mkdir(); sentinel = self.target / "keep"; sentinel.write_text("owner data")
        with self.assertRaises(ValueError): self.run_import()
        self.assertEqual(sentinel.read_text(), "owner data")

    def test_metadata_complexity_dtype_and_aggregate_limits(self):
        from hyperl.safetensors_import import _json
        for data in (b'{"x":' + b'[' * 33 + b'0' + b']' * 33 + b'}',
                     b'{"x":1e999}', b'{"x":' + b'9' * 21 + b'}'):
            with self.assertRaises(ValueError): _json(data)
        path = self.source / "model.safetensors"
        self.write_shard(path.name, self.weights)
        raw = path.read_bytes(); length = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8 + length]); header["model.norm.weight"]["dtype"] = []
        encoded = json.dumps(header).encode(); path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8 + length:])
        with self.assertRaises(ValueError): self.run_import()
        self.write_shard(path.name, self.weights)
        with patch("hyperl.safetensors_import.AGGREGATE_HEADER_LIMIT", 32):
            with self.assertRaises(ValueError): self.run_import()
        self.assert_no_output()

    def test_source_mutation_during_conversion_cannot_publish(self):
        self.write_shard("model.safetensors", self.weights)
        mutated = False
        def change_source():
            nonlocal mutated
            if not mutated and list(self.root.glob(".hyperl-import-*")):
                mutated = True
                self.cfg["rope_theta"] = 20000.0
                self.write_config()
            return False
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.run_import(cancelled=change_source)
        self.assertTrue(mutated)
        self.assert_no_output()

    def hf_reference(self, weights, tokens):
        # Independent HF-shaped full-prefix forward; no converter or JAX operations.
        np = self.np; x = weights["model.embed_tokens.weight"][tokens].copy(); n = len(tokens)
        def norm(v, w): return v / np.sqrt(np.mean(v * v, axis=-1, keepdims=True) + 1e-5) * w
        angle = np.arange(n, dtype=np.float32)[:, None] / (10000.0 ** (np.arange(0, 4, 2, dtype=np.float32)[None, :] / 4))
        angle = np.concatenate((angle, angle), axis=-1)[:, None, :]
        def rope(v): return v * np.cos(angle) + np.concatenate((-v[..., 2:], v[..., :2]), axis=-1) * np.sin(angle)
        for i in range(2):
            p = f"model.layers.{i}."; a = norm(x, weights[p + "input_layernorm.weight"])
            q = rope((a @ weights[p + "self_attn.q_proj.weight"].T).reshape(n, 4, 4))
            k = rope((a @ weights[p + "self_attn.k_proj.weight"].T).reshape(n, 2, 4)).repeat(2, axis=1)
            v = (a @ weights[p + "self_attn.v_proj.weight"].T).reshape(n, 2, 4).repeat(2, axis=1)
            scores = np.einsum("qhd,khd->hqk", q, k) / 2
            scores = np.where(np.tril(np.ones((n, n), dtype=bool))[None], scores, -np.inf)
            probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True)); probabilities /= probabilities.sum(axis=-1, keepdims=True)
            x += np.einsum("hqk,khd->qhd", probabilities, v).reshape(n, 16) @ weights[p + "self_attn.o_proj.weight"].T
            a = norm(x, weights[p + "post_attention_layernorm.weight"])
            gate = a @ weights[p + "mlp.gate_proj.weight"].T
            x += ((gate / (1 + np.exp(-gate))) * (a @ weights[p + "mlp.up_proj.weight"].T)) @ weights[p + "mlp.down_proj.weight"].T
        return norm(x, weights["model.norm.weight"]) @ weights["lm_head.weight"].T

    @unittest.skipUnless(HAS_JAX, "explicit optional JAX environment required; no execution qualified")
    def test_real_jax_logits_and_tokens_match_independent_hf_layout_reference(self):
        from hyperl.jax_llm import JaxDecoder
        for dtype in ("F32", "F16", "BF16"):
            with self.subTest(dtype=dtype):
                self.target = self.root / dtype
                weights = self.write_shard("model.safetensors", self.weights, dtype)
                result = self.run_import()
                decoder = JaxDecoder.load(self.target, manifest_sha256=result["manifest_sha256"], backend="cpu", device_ids=[0], budget_bytes=4 * 1024 * 1024)
                tokens = [2, 7, 3]; expected = []
                for _ in range(3):
                    token = int(self.hf_reference(weights, tokens)[-1].argmax()); expected.append(token); tokens.append(token)
                actual = decoder.generate([2, 7, 3], max_tokens=3)
                self.assertEqual(actual["tokens"], expected)
                self.np.testing.assert_allclose(self.np.asarray(decoder.logits), self.hf_reference(weights, tokens)[-1], rtol=3e-5, atol=3e-5)


if __name__ == "__main__": unittest.main()
