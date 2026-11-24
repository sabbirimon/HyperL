#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
"""Synthetic durable queue/claim benchmark; no live devices or model execution."""
import argparse
import json
from pathlib import Path
import resource
import secrets
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from hyperl.scheduler import Cell, FleetScheduler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--jobs", type=int, default=1000000)
    parser.add_argument("--claims", type=int, default=1000)
    args = parser.parse_args()
    if args.database.exists() or not 1 <= args.claims <= args.jobs <= 10_000_000:
        parser.error("new database and 1<=claims<=jobs<=10000000 required")
    scheduler = FleetScheduler(args.database, max_jobs=args.jobs, max_pending_per_tenant=args.jobs, audit_key=secrets.token_bytes(32))
    try:
        scheduler.register(Cell("cell-a", "instance-a", 1, ("combined",), ("model-a",)), now=1000, ttl_ms=300000)
        started = time.perf_counter()
        for start in range(0, args.jobs, 1000):
            scheduler.submit("tenant-a", (dict(key=f"job-{i:09}", pool="combined", profile="model-a", payload={"prompt": "synthetic", "max_tokens": 1})
                             for i in range(start, min(args.jobs, start + 1000))), now=1000)
        inserted = time.perf_counter() - started
        started = time.perf_counter()
        for _ in range(args.claims):
            receipt = scheduler.claim("cell-a", "instance-a", 1, now=1001)
            if receipt is None:
                raise RuntimeError("missing indexed queue job")
            scheduler.start(receipt, now=1001)
            scheduler.finish(receipt, {"synthetic": True}, now=1001)
        dispatched = time.perf_counter() - started
        states = {row[0]: row[1] for row in scheduler.db.execute("SELECT state,count(*) FROM jobs GROUP BY state")}
        if states.get("SUCCEEDED", 0) != args.claims or sum(states.values()) != args.jobs:
            raise RuntimeError("durable queue accounting mismatch")
        scheduler.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(json.dumps(dict(synthetic=True, live_devices=0, jobs=args.jobs, claimed_started_finished=args.claims,
            insert_seconds=inserted, dispatch_seconds=dispatched, states=states, database_bytes=args.database.stat().st_size, audit_head=scheduler.audit.verify(),
            peak_process_rss_bytes=rss if sys.platform == "darwin" else rss * 1024), indent=2))
    finally:
        scheduler.close()


if __name__ == "__main__":
    main()
