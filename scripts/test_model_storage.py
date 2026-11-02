# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
import dataclasses
import pathlib
import sys
import tempfile
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
from hyperl.model_storage import ModelStore, ModelChunkCache, ModelLayer, MemoryTier, Workload, LLMProfile, estimate_llm, plan_model


class ModelStorageTest(unittest.TestCase):
    def test_streaming_import_dedup_export_and_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "weights.bin"
            source.write_bytes(b"a" * 8192 + b"b" * 4096 + b"tail")
            store = ModelStore(root / "store", 4096)
            try:
                first = store.ingest(source, max_bytes=20000)
                second = store.ingest(source, max_bytes=20000)
                self.assertNotEqual(first, second)
                self.assertEqual(store.info(first)["bytes"], source.stat().st_size)
                self.assertEqual(len(list((root / "store" / "blobs").glob("*/*"))), 3)
                destination = root / "export.bin"
                store.export(first, destination, max_bytes=20000)
                self.assertEqual(destination.read_bytes(), source.read_bytes())
                with self.assertRaises(ValueError):
                    store.export(first, destination, max_bytes=20000)
            finally:
                store.close()
            store = ModelStore(root / "store")
            try:
                self.assertEqual(b"".join(store.iter_chunks(second, max_bytes=20000)), source.read_bytes())
            finally:
                store.close()

    def test_tamper_quota_and_cancel_never_publish_partial_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "weights.bin"
            source.write_bytes(b"a" * 9000)
            store = ModelStore(root / "store", 4096)
            try:
                with self.assertRaises(ValueError):
                    store.ingest(source, max_bytes=8000)
                with self.assertRaises(InterruptedError):
                    store.ingest(source, max_bytes=10000, cancelled=lambda: True)
                self.assertEqual(store.db.execute("SELECT count(*) FROM artifacts").fetchone()[0], 0)
                artifact = store.ingest(source, max_bytes=10000)
                with self.assertRaises(InterruptedError):
                    store.export(artifact, root / "cancelled", max_bytes=10000, cancelled=lambda: True)
                blob = next((root / "store" / "blobs").glob("*/*"))
                blob.write_bytes(b"x" * blob.stat().st_size)
                with self.assertRaises(ValueError):
                    store.export(artifact, root / "corrupt", max_bytes=10000)
                self.assertFalse((root / "cancelled").exists())
                self.assertFalse((root / "corrupt").exists())
                self.assertFalse(list(root.glob(".hyperl-export-*")))
            finally:
                store.close()

    def test_manifest_order_and_total_hash_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "weights"
            source.write_bytes(b"a" * 5000)
            store = ModelStore(root / "store", 4096)
            try:
                artifact = store.ingest(source, max_bytes=5000)
                store.db.execute("UPDATE artifacts SET sha256=? WHERE id=?", ("0" * 64, artifact))
                store.db.commit()
                with self.assertRaises(ValueError):
                    store.export(artifact, root / "export", max_bytes=5000)
                self.assertFalse((root / "export").exists())
            finally:
                store.close()

    def test_ram_cache_has_hard_payload_bound_and_lru_eviction(self):
        cache = ModelChunkCache(8)
        a = cache.remember(b"aaaa")
        b = cache.remember(b"bbbb")
        self.assertEqual(cache.get(a), b"aaaa")
        cache.remember(b"cccc")
        self.assertIsNone(cache.get(b))
        self.assertLessEqual(cache.used, 8)
        cache.remember(b"x" * 100)
        self.assertLessEqual(cache.used, 8)


class ModelPlannerTest(unittest.TestCase):
    def tiers(self):
        return [MemoryTier("gpu", "a", "GPU", 1000, 1000., "vram-a", "test-model/1"),
                MemoryTier("ram", "a", "RAM", 2000, 100., "ram-a"),
                MemoryTier("disk", "a", "STORAGE", 10000, 10., "disk-a")]

    def test_auto_scenarios_and_manual_override(self):
        layers = [ModelLayer("layer", 100, 10)]
        for policy, expected in [(Workload(), "decode"), (Workload(prompt_tokens=8192), "prefill"),
                                 (Workload(concurrent_requests=8), "batch"), (Workload(training=True), "training"),
                                 (Workload(scenario="prefill"), "prefill")]:
            plan = plan_model(layers, self.tiers(), policy)
            self.assertEqual(plan["scenario"], expected)
            self.assertFalse(plan["execution_ready"])
        with self.assertRaises(ValueError):
            plan_model(layers, self.tiers(), Workload(training=True, scenario="decode"))

    def test_ram_and_disk_offload_reserve_staging_and_limit_transfer(self):
        layers = [ModelLayer("a", 600), ModelLayer("b", 600)]
        plan = plan_model(layers, self.tiers(), Workload(reserve_fraction=0), manual={"a": "ram", "b": "ram"})
        self.assertEqual(plan["staging_bytes"]["gpu"], 600)
        self.assertEqual(plan["used_bytes"]["ram"], 1200)
        self.assertEqual(plan["estimated_weight_transfer_seconds_per_pass"], 12.)
        plan = plan_model([ModelLayer("a", 600)], self.tiers(), Workload(reserve_fraction=0, allow_storage_offload=True), manual={"a": "disk"})
        self.assertEqual(plan["staging_bytes"]["gpu"], 600)
        with self.assertRaises(ValueError):
            plan_model(layers, self.tiers(), Workload(max_transfer_seconds=1), manual={"a": "ram", "b": "ram"})
        with self.assertRaises(ValueError):
            plan_model(layers, self.tiers(), Workload(), manual={"a": "disk"})

    def test_capacity_aliases_incompatible_backends_and_state_fail_closed(self):
        tiers = self.tiers()
        with self.assertRaises(ValueError):
            plan_model([ModelLayer("a", 100)], tiers + [dataclasses.replace(tiers[0], id="alias")])
        with self.assertRaises(ValueError):
            plan_model([ModelLayer("a", 100)], tiers + [MemoryTier("tpu", "b", "TPU", 1000, 100., "hbm-b", "different-runtime")])
        with self.assertRaises(ValueError):
            plan_model([ModelLayer("a", 100, 10000)], tiers)
        with self.assertRaises(ValueError):
            plan_model([ModelLayer("huge", 30000)], tiers)
        with self.assertRaises(ValueError):
            plan_model([ModelLayer("a", 100)], [tiers[-1]])

    def test_llm_estimator_accounts_quantization_context_gqa_and_workspace(self):
        profile = LLMProfile(hidden_size=4096, intermediate_size=11008, layers=32, attention_heads=32, kv_heads=8, vocabulary=32000)
        layers, activation = estimate_llm(profile, Workload(prompt_tokens=4096, max_new_tokens=512, concurrent_requests=2))
        self.assertEqual(len(layers), 33)
        self.assertEqual(layers[1].kv_bytes, 2 * 8 * 128 * 4608 * 2 * 2)
        quantized, _ = estimate_llm(dataclasses.replace(profile, weight_bits=4))
        self.assertLess(quantized[1].weight_bytes, layers[1].weight_bytes)
        _, linear = estimate_llm(dataclasses.replace(profile, attention_workspace="linear"), Workload(prompt_tokens=4096, concurrent_requests=2))
        self.assertGreater(activation, linear)
        with self.assertRaises(ValueError):
            estimate_llm(dataclasses.replace(profile, attention_heads=31))

    def test_multiple_accelerators_partition_whole_layers_and_account_every_byte(self):
        tiers = [MemoryTier("a", "a", "GPU", 1000, 1000., "a", "model/1"), MemoryTier("b", "b", "GPU", 1000, 500., "b", "model/1")]
        plan = plan_model([ModelLayer("a", 700, 100), ModelLayer("b", 700, 100)], tiers, Workload(reserve_fraction=0))
        self.assertEqual({p["execution_tier"] for p in plan["placements"]}, {"a", "b"})
        self.assertEqual(sum(plan["used_bytes"].values()), 1600)
        self.assertTrue(all(plan["used_bytes"][t.id] <= t.available_bytes for t in tiers))


if __name__ == "__main__":
    unittest.main()
