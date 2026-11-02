#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
"""Synthetic streaming artifact check; creates and cleans only a private temporary tree."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import tempfile
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from hyperl.model_storage import ModelStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path, help="existing private directory with source/store/export headroom")
    parser.add_argument("--mib", type=int, default=256)
    args = parser.parse_args()
    if not args.workdir.is_absolute() or not args.workdir.is_dir() or not 1 <= args.mib <= 16384:
        parser.error("absolute existing work directory and 1..16384 MiB required")
    with tempfile.TemporaryDirectory(prefix="hyperl-model-bench-", dir=args.workdir) as directory:
        root = Path(directory)
        source = root / "synthetic-weights.bin"
        expected = hashlib.sha256()
        with source.open("wb") as stream:
            for i in range(args.mib):
                seed = hashlib.sha256(str(i).encode()).digest()
                block = seed * (1024 * 1024 // len(seed))
                stream.write(block)
                expected.update(block)
        store = ModelStore(root / "store")
        try:
            start = time.perf_counter()
            artifact = store.ingest(source, max_bytes=args.mib * 1024 * 1024)
            imported = time.perf_counter() - start
            info = store.info(artifact)
            if info["sha256"] != expected.hexdigest():
                raise RuntimeError("import digest mismatch")
            start = time.perf_counter()
            restored = root / "restored.bin"
            store.export(artifact, restored, max_bytes=args.mib * 1024 * 1024)
            exported = time.perf_counter() - start
            actual = hashlib.sha256()
            with restored.open("rb") as stream:
                while block := stream.read(4 * 1024 * 1024):
                    actual.update(block)
            if actual.hexdigest() != expected.hexdigest() or restored.stat().st_size != info["bytes"]:
                raise RuntimeError("export digest/size mismatch")
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            print(json.dumps(dict(synthetic=True, llm_inference=False, bytes=info["bytes"], chunks=info["chunks"],
                                  import_seconds=imported, export_seconds=exported, whole_artifact_verified=True,
                                  peak_process_rss_bytes=rss if sys.platform == "darwin" else rss * 1024), indent=2))
        finally:
            store.close()


if __name__ == "__main__":
    main()
