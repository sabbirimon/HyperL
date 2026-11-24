#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
"""Synthetic disk inventory benchmark. No network connections or live device claims."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from hyperl.fleet import Device, DeviceDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, help="new private local SQLite path")
    parser.add_argument("--devices", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.database.exists() or not 1 <= args.devices <= 10_000_000:
        parser.error("new database and 1..10000000 synthetic records required")
    directory = DeviceDirectory(args.database)
    now = 1000
    started = time.perf_counter()
    try:
        for start in range(0, args.devices, 1000):
            directory.enroll((Device(f"node-{i:09}", f"cell-{i // 16:08}", "arm64" if i % 2 else "x86_64",
                                     f"https://node-{i}.invalid:8443", f"instance-{i}", 1, now + 300000)
                              for i in range(start, min(args.devices, start + 1000))), now)
        imported = time.perf_counter() - started
        counted = 0
        after = ""
        read_started = time.perf_counter()
        while True:
            page = directory.page(now + 1, after=after, limit=1000)
            if not page:
                break
            counted += len(page)
            after = page[-1].id
        paged = time.perf_counter() - read_started
        if counted != args.devices:
            raise RuntimeError("inventory count mismatch")
        cell = directory.page(now + 1, cell="cell-00000000")
        if len(cell) != min(16, args.devices) or directory.page(now + 300000):
            raise RuntimeError("cell/expiry mismatch")
        directory.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = rss if sys.platform == "darwin" else rss * 1024
        print(json.dumps(dict(synthetic=True, live_devices=0, devices=counted, batch_size=1000, page_size=1000,
                              insert_seconds=imported, page_all_seconds=paged, database_bytes=args.database.stat().st_size,
                              peak_process_rss_bytes=peak_bytes, cell_size=len(cell), expiry_checked=True), indent=2))
    finally:
        directory.close()


if __name__ == "__main__":
    main()
