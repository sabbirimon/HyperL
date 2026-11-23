"""Index manuscript pointers from the pinned upstream catalog, entirely offline."""
import argparse
import csv
import hashlib
import re
from pathlib import Path

REVISION = "4eaee87548250b2682affda7d05d947c05554bae"
SOURCE_SHA = "39de58f1bf2185eb11d444a58aec4c7c7530419f3127deb9261fa46e044a4135"
PREFIX = "https://github.com/openai/math/blob/main/preprints/"

def index(path, output):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError("source digest differs from reviewed catalog")
    subject = None
    rows = []
    for line in raw.decode("utf-8").splitlines():
        if line.startswith(r"\cataloguesection{"):
            subject = line.split("{", 1)[1].split("}", 1)[0]
        if line.startswith(r"\resultentry{"):
            family = line.split("{", 1)[1].split("}", 1)[0]
            for url in re.findall(r"\\href\{([^}]+)\}", line):
                if url.startswith(PREFIX) and url.endswith(".pdf"):
                    pinned = url.replace("/blob/main/", f"/blob/{REVISION}/", 1)
                    rows.append((family, subject, pinned))
    if len({r[0] for r in rows}) != 373 or len({r[2] for r in rows}) != len(rows) or len(rows) != 721:
        raise ValueError("unexpected family or manuscript count")
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("family_id", "subject", "manuscript_url"))
        writer.writerows(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    index(args.catalog, args.output)
