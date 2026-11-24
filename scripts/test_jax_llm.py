# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
import importlib.util
import pathlib
import sys
import tempfile
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
from hyperl.jax_llm import DecoderConfig, JaxDecoder, JaxSettings, JaxModelSession
from hyperl.llm import file_digest
import json


class DecoderContractTest(unittest.TestCase):
    def test_invalid_shapes_reject_before_loading_jax(self):
        for cfg in [DecoderConfig(32, 15, 32, 2, 4, 4), DecoderConfig(32, 16, 32, 2, 4, 3), DecoderConfig(0, 16, 32, 2, 4, 4)]:
            with self.assertRaises(ValueError):
                cfg.validate()
        self.assertEqual(DecoderConfig(32, 16, 32, 2, 4, 2).shapes()["0.k"], (16, 8))


@unittest.skipUnless(importlib.util.find_spec("jax") is not None, "explicit optional JAX test environment not installed")
class JaxExecutionTest(unittest.TestCase):
    def fixture(self):
        import numpy as np
        cfg = DecoderConfig(32, 16, 32, 2, 4, 4, context=32)
        rng = np.random.default_rng(42)
        weights = {name: (np.ones(shape, np.float32) if "norm" in name else rng.normal(0, .07, shape).astype(np.float32)) for name, shape in cfg.shapes().items()}
        return cfg, weights

    def reference(self, cfg, weights, tokens):
        import numpy as np
        # Independent full-prefix attention, without a KV cache or JAX primitives.
        x = weights["embedding"][tokens].copy()
        d = cfg.hidden // cfg.heads
        def norm(value, scale):
            return value / np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + cfg.rms_epsilon) * scale
        def rotary(value):
            angle = np.arange(len(tokens), dtype=np.float32)[:, None] / (cfg.rope_theta ** (np.arange(0, d, 2, dtype=np.float32)[None, :] / d))
            angles = np.concatenate([angle, angle], axis=-1)[:, None, :]
            rotated = np.concatenate([-value[..., d // 2:], value[..., :d // 2]], axis=-1)
            return value * np.cos(angles) + rotated * np.sin(angles)
        for i in range(cfg.layers):
            prefix = str(i) + "."
            a = norm(x, weights[prefix + "attention_norm"])
            q = rotary((a @ weights[prefix + "q"]).reshape(-1, cfg.heads, d))
            k = rotary((a @ weights[prefix + "k"]).reshape(-1, cfg.kv_heads, d))
            v = (a @ weights[prefix + "v"]).reshape(-1, cfg.kv_heads, d)
            k = np.repeat(k, cfg.heads // cfg.kv_heads, axis=1)
            v = np.repeat(v, cfg.heads // cfg.kv_heads, axis=1)
            scores = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(np.float32(d))
            scores = np.where(np.tril(np.ones((len(tokens), len(tokens)), dtype=bool))[None], scores, -np.inf)
            probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            attention = np.einsum("hqk,khd->qhd", probabilities, v).reshape(-1, cfg.hidden)
            x += attention @ weights[prefix + "o"]
            a = norm(x, weights[prefix + "ffn_norm"])
            gate = a @ weights[prefix + "gate"]
            x += ((gate / (1 + np.exp(-gate))) * (a @ weights[prefix + "up"])) @ weights[prefix + "down"]
        return norm(x, weights["norm"]) @ weights["head"]

    def test_real_decoder_logits_and_generation_match_independent_reference(self):
        import numpy as np
        cfg, weights = self.fixture()
        decoder = JaxDecoder(cfg, weights, backend="cpu", device_ids=[0], budget_bytes=16 * 1024 * 1024)
        tokens = [1, 2, 3]
        result = decoder.generate(tokens, max_tokens=4)
        expected = []
        for _ in range(4):
            next_id = int(self.reference(cfg, weights, tokens)[-1].argmax())
            expected.append(next_id); tokens.append(next_id)
        self.assertEqual(result["tokens"], expected)
        np.testing.assert_allclose(np.asarray(decoder.logits), self.reference(cfg, weights, tokens)[-1], rtol=2e-5, atol=2e-5)
        self.assertEqual(decoder.position, 7)
        self.assertFalse(result["physical_tpu_tested"])

    def test_four_device_weight_and_kv_shards_match_single_device(self):
        import jax
        import numpy as np
        if len(jax.devices("cpu")) < 4:
            self.skipTest("run with XLA_FLAGS=--xla_force_host_platform_device_count=4")
        cfg, weights = self.fixture()
        single = JaxDecoder(cfg, weights, backend="cpu", device_ids=[0], budget_bytes=16 * 1024 * 1024)
        sharded = JaxDecoder(cfg, weights, backend="cpu", device_ids=[0, 1, 2, 3], budget_bytes=16 * 1024 * 1024)
        a = single.generate([1, 2, 3], max_tokens=4)
        b = sharded.generate([1, 2, 3], max_tokens=4)
        self.assertEqual(a["tokens"], b["tokens"])
        np.testing.assert_allclose(np.asarray(single.logits), np.asarray(sharded.logits), rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(np.asarray(single.keys), np.asarray(sharded.keys), rtol=2e-5, atol=2e-5)
        self.assertEqual(len(sharded.keys.addressable_shards), 4)
        self.assertEqual(sharded.keys.addressable_shards[0].data.shape[2], 1)
        self.assertFalse(sharded.weights["0.q"].is_fully_replicated)
        self.assertTrue(sharded.weights["norm"].is_fully_replicated)

    def test_checked_token_boundary_checkpoint_resumes_and_rejects_tamper(self):
        import numpy as np
        cfg, weights = self.fixture()
        first = JaxDecoder(cfg, weights, backend="cpu", device_ids=[0], budget_bytes=16 * 1024 * 1024)
        first.generate([1, 2, 3], max_tokens=2)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            saved = first.save_checkpoint(root / "checkpoint", max_bytes=1024 * 1024)
            expected = first.generate([], max_tokens=3, reset=False)
            restored = JaxDecoder(cfg, weights, backend="cpu", device_ids=[0], budget_bytes=16 * 1024 * 1024)
            restored.restore_checkpoint(root / "checkpoint", manifest_sha256=saved["manifest_sha256"], max_bytes=1024 * 1024)
            self.assertEqual(restored.position, saved["position"])
            actual = restored.generate([], max_tokens=3, reset=False)
            self.assertEqual(actual["tokens"], expected["tokens"])
            np.testing.assert_allclose(np.asarray(first.logits), np.asarray(restored.logits), atol=2e-5, rtol=2e-5)
            (root / "checkpoint" / "keys-0.npy").write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                restored.restore_checkpoint(root / "checkpoint", manifest_sha256=saved["manifest_sha256"], max_bytes=1024 * 1024)
            with self.assertRaises(InterruptedError):
                first.save_checkpoint(root / "cancelled", max_bytes=1024 * 1024, cancelled=lambda: True)
            self.assertFalse((root / "cancelled").exists())

    def test_pinned_model_file_loader_executes_through_the_public_profile(self):
        import numpy as np
        cfg, weights = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary); records = {}
            for name, value in weights.items():
                target = root / (name + ".npy")
                np.save(target, value, allow_pickle=False)
                records[name] = dict(file=target.name, sha256=file_digest(target))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(format="hyperl-llama-f32/1", config=cfg.__dict__, tensors=records)))
            settings = JaxSettings(str(root), file_digest(manifest), "cpu", (0,), budget_bytes=16 * 1024 * 1024)
            session = JaxModelSession(settings)
            try:
                result = session.complete([1, 2, 3], max_tokens=2)
                self.assertEqual(len(result["tokens"]), 2)
                self.assertEqual(result["checkpoint_manifest_sha256"], settings.manifest_sha256)
            finally:
                session.close()
            (root / "0.q.npy").write_bytes(b"wrong")
            with self.assertRaises(ValueError):
                JaxModelSession(settings)

    def test_unavailable_tpu_memory_mesh_cancel_and_context_reject(self):
        cfg, weights = self.fixture()
        with self.assertRaises(RuntimeError):
            JaxDecoder(cfg, weights, backend="tpu", device_ids=[0])
        with self.assertRaises(ValueError):
            JaxDecoder(cfg, weights, backend="cpu", device_ids=[0], budget_bytes=1024)
        decoder = JaxDecoder(cfg, weights, backend="cpu", device_ids=[0], budget_bytes=16 * 1024 * 1024)
        with self.assertRaises(InterruptedError):
            decoder.generate([1, 2], max_tokens=1, cancelled=lambda: True)
        self.assertTrue(decoder.failed)
        with self.assertRaises(ValueError):
            decoder.generate([1] * 30, max_tokens=4)


if __name__ == "__main__":
    unittest.main()
