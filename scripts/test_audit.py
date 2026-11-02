# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from hyperl.scheduler import FleetScheduler, Cell
from hyperl.audit import create_key, read_key


class AuditTest(unittest.TestCase):
    def test_external_key_signed_transactions_reopen_and_no_downgrade(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); key_file = root / "audit.key"; create_key(key_file)
            key = read_key(key_file); path = root / "ledger"
            scheduler = FleetScheduler(path, audit_key=key)
            scheduler.register(Cell("cell", "instance", 1, ("pool",), ("model",)), now=1000)
            scheduler.submit("tenant", [dict(key="job", pool="pool", profile="model", payload={"prompt": "private-sentinel"})], now=1000)
            head = scheduler.audit.verify()
            self.assertEqual(head["algorithm"], "HMAC-SHA256/1")
            self.assertNotIn("private-sentinel", str(scheduler.audit.page()))
            scheduler.close()
            with self.assertRaises(ValueError):
                FleetScheduler(path)
            with self.assertRaises(ValueError):
                FleetScheduler(path, audit_key=b"wrong" * 6 + b"xx")
            reopened = FleetScheduler(path, audit_key=key)
            try:
                self.assertEqual(reopened.audit.verify(expected_head=head), head)
                with self.assertRaises(Exception):
                    reopened.db.execute("DELETE FROM audit_events")
                reopened.db.rollback()
                # Simulate an administrator bypassing ordinary append-only SQL guards.
                reopened.db.execute("DROP TRIGGER audit_no_update")
                reopened.db.execute("UPDATE audit_events SET kind='tampered' WHERE seq=1"); reopened.db.commit()
                with self.assertRaises(ValueError):
                    reopened.audit.verify()
            finally:
                reopened.close()

    def test_audit_quota_exhaustion_rolls_back_new_authority(self):
        scheduler = FleetScheduler(":memory:", audit_key=b"x" * 32)
        try:
            scheduler.audit.max_bytes = 1024
            before = scheduler.audit.head()
            with self.assertRaises(ValueError):
                scheduler.submit("tenant", [dict(key=f"job-{i}", pool="pool", profile="model", payload={"prompt": "a"}) for i in range(20)], now=1000)
            self.assertEqual(scheduler.audit.head(), before)
            self.assertEqual(scheduler.page("tenant"), [])
        finally:
            scheduler.close()

    def test_cancelled_batch_rolls_back_its_audit_and_state(self):
        scheduler = FleetScheduler(":memory:", audit_key=b"x" * 32)
        try:
            before = scheduler.audit.head()
            request = dict(key="same", pool="pool", profile="model", payload={"prompt": "a"})
            bad = dict(request, payload={"prompt": "b"})
            with self.assertRaises(ValueError):
                scheduler.submit("tenant", [request, bad], now=1000)
            self.assertEqual(scheduler.audit.head(), before)
            self.assertEqual(scheduler.page("tenant"), [])
        finally:
            scheduler.close()


if __name__ == "__main__":
    unittest.main()
