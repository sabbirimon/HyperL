# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
import itertools
import os
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from hyperl.erasure import ErasureCodec, ProtectedModelStore
from hyperl.model_storage import ModelStore
from hyperl.llm import file_digest


def library():
    explicit = os.environ.get("HYPERL_ERASURE_LIBRARY")
    if explicit:
        return Path(explicit).resolve()
    matches = [p.resolve() for p in Path(__file__).resolve().parents[1].joinpath("build/native").rglob("*hyperl_erasure*") if p.suffix in (".dylib", ".so", ".dll")]
    return matches[0] if len(matches) == 1 else None


@unittest.skipUnless(library(), "build the explicit native erasure library first")
class ErasureTest(unittest.TestCase):
    def codec(self):
        path = library()
        return ErasureCodec(str(path), file_digest(path))

    def test_every_two_fragment_erasure_and_corrupt_hash_exclusion(self):
        codec = self.codec()
        data = [bytes((i * 51 + j) % 256 for j in range(1000)) for i in range(4)]
        all_fragments = data + codec.encode(data, 2)
        for missing in itertools.combinations(range(6), 2):
            damaged = [None if i in missing else value for i, value in enumerate(all_fragments)]
            self.assertEqual(codec.recover(damaged, 4, 2), all_fragments)
        with self.assertRaises(ValueError):
            codec.recover([None, None, None, *all_fragments[3:]], 4, 2)
        with self.assertRaises(InterruptedError):
            codec.encode(data, 2, cancelled=lambda: True)
        with self.assertRaises(ValueError):
            codec.encode([b"a", b"bb"], 2)

    def test_model_store_repairs_missing_data_and_parity_then_checks_whole_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / "weights"
            source.write_bytes(b"".join(bytes([i]) * 4096 for i in range(1, 6)) + b"tail")
            store = ModelStore(root / "store", 4096)
            protected = ProtectedModelStore(store, self.codec())
            try:
                identifier = store.ingest(source, max_bytes=30000)
                manifest = protected.protect(identifier, max_bytes=30000)
                data = list(store.db.execute("SELECT digest FROM chunks WHERE artifact=? ORDER BY ordinal", (identifier,)))
                for row in data[:2]:
                    key = row[0]; (store.blobs / key[:2] / key).unlink()
                # One additional erasure on a different stripe.
                key = data[-1][0]; (store.blobs / key[:2] / key).write_bytes(b"bad")
                with self.assertRaises(ValueError):
                    store.export(identifier, root / "partial", max_bytes=30000)
                report = protected.repair(identifier, expected_manifest_sha256=manifest["manifest_sha256"], max_bytes=30000)
                self.assertTrue(report["whole_artifact_verified"]); self.assertGreaterEqual(report["repaired_fragments"], 3)
                store.export(identifier, root / "restored", max_bytes=30000)
                self.assertEqual((root / "restored").read_bytes(), source.read_bytes())
                self.assertFalse((root / "partial").exists())
            finally:
                store.close()

    def test_manifest_tamper_and_beyond_budget_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / "weights"
            source.write_bytes(b"".join(bytes([i]) * 4096 for i in range(1, 5)))
            store = ModelStore(root / "store", 4096); protected = ProtectedModelStore(store, self.codec())
            try:
                identifier = store.ingest(source, max_bytes=20000)
                manifest = protected.protect(identifier, max_bytes=20000)
                for key, in list(store.db.execute("SELECT digest FROM chunks WHERE artifact=?", (identifier,)))[:3]:
                    (store.blobs / key[:2] / key).unlink()
                with self.assertRaises(ValueError):
                    protected.repair(identifier, expected_manifest_sha256=manifest["manifest_sha256"], max_bytes=20000)
                store.db.execute("UPDATE protection_groups SET metadata=replace(metadata,'4096','4095') WHERE artifact=?", (identifier,)); store.db.commit()
                with self.assertRaises(ValueError):
                    protected.repair(identifier, expected_manifest_sha256=manifest["manifest_sha256"], max_bytes=20000)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
