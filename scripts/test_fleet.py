# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
import dataclasses
import pathlib
import sys
import tempfile
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
from hyperl.fleet import Device, DeviceDirectory, shard_for


class FleetTest(unittest.TestCase):
    def device(self, i, **changes):
        d = Device(f"node-{i:08}", "cell-a" if i % 2 else "cell-b", "arm64", "https://node.example:8443", f"instance-{i}", 1, 20000)
        return dataclasses.replace(d, **changes)

    def test_persistent_keyset_pages_named_pools_and_expiry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "fleet.db"
            db = DeviceDirectory(path)
            try:
                db.enroll([self.device(i) for i in range(1000)], 1000)
                db.enroll([self.device(i) for i in range(1000, 2000)], 1000)
                db.set_pool("combined", ["cell-a", "cell-b"])
                seen = []
                after = ""
                while True:
                    page = db.page(1001, pool="combined", after=after, limit=137)
                    if not page:
                        break
                    seen.extend(d.id for d in page)
                    after = page[-1].id
                self.assertEqual(len(seen), 2000)
                self.assertEqual(len(set(seen)), 2000)
                self.assertTrue(all(d.cell == "cell-a" for d in db.page(1001, cell="cell-a")))
                self.assertEqual(db.page(20000), [])
            finally:
                db.close()
            db = DeviceDirectory(path)
            try:
                self.assertEqual(db.resolve("node-00000001", 1, 1001).instance, "instance-1")
            finally:
                db.close()

    def test_revocation_restart_and_lease_fences_are_atomic(self):
        db = DeviceDirectory(":memory:")
        try:
            d = self.device(1)
            db.enroll([d], 1000)
            self.assertTrue(db.revoke(d.id, 1))
            with self.assertRaises(ValueError):
                db.enroll([d], 1000)
            with self.assertRaises(ValueError):
                db.resolve(d.id, 1, 1000)
            restarted = dataclasses.replace(d, epoch=2, instance="new-instance")
            db.enroll([restarted], 1000)
            with self.assertRaises(ValueError):
                db.enroll([self.device(2), d], 1000)
            with self.assertRaises(ValueError):
                db.resolve(self.device(2).id, 1, 1000)
            self.assertEqual(db.resolve(d.id, 2, 1000), restarted)
            with self.assertRaises(ValueError):
                db.enroll([dataclasses.replace(restarted, endpoint="https://other.example")], 1000)
        finally:
            db.close()

    def test_remote_security_bounds_and_stable_sharding(self):
        db = DeviceDirectory(":memory:")
        try:
            for endpoint in ["http://192.168.0.1", "http://localhost", "https://u:p@host", "https://host/path", "https://host?token=secret"]:
                with self.assertRaises(ValueError):
                    db.enroll([self.device(1, endpoint=endpoint)], 1000)
            with self.assertRaises(ValueError):
                db.enroll([self.device(1, expires_ms=301001)], 1000)
            with self.assertRaises(ValueError):
                db.enroll((self.device(i) for i in range(1001)), 1000)
            with self.assertRaises(ValueError):
                db.page(1000, limit=1001)
            self.assertEqual(shard_for("node-1", 1024), shard_for("node-1", 1024))
            self.assertTrue(0 <= shard_for("node-2", 1024) < 1024)
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
