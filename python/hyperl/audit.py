# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
# Copyright 2026 Sabbir Hassan Imon and respective contributors.
"""Transactional hash-linked audit records; optional external HMAC authority.

Preserve checkpoints outside the database to detect suffix truncation. A local key
and chain do not provide WORM storage, nonrepudiation or protection from host root.
"""
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import stat


def create_key(path):
    path = Path(path)
    if not path.is_absolute() or not path.parent.is_dir():
        raise ValueError("new absolute private audit key path required")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(secrets.token_bytes(32))


def read_key(path):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.stat().st_size != 32:
        raise ValueError("private absolute 32-byte audit key required")
    if os.name != "nt" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError("audit key permissions must exclude group/other access")
    return path.read_bytes()


class AuditTrail:
    def __init__(self, db, key=None, *, max_bytes=1024**3):
        if key is not None and (not isinstance(key, bytes) or len(key) != 32):
            raise ValueError("audit key must be exactly 32 bytes")
        if type(max_bytes) is not int or not 1024 <= max_bytes <= 1024**4:
            raise ValueError("bounded audit storage quota required")
        self.db = db; self.key = key; self.max_bytes = max_bytes
        self.algorithm = "HMAC-SHA256/1" if key is not None else "SHA256-laboratory/1"
        self.key_id = hashlib.sha256(key).hexdigest() if key is not None else "none"
        db.executescript("""
            CREATE TABLE IF NOT EXISTS audit_budget (id INTEGER PRIMARY KEY CHECK(id=1), bytes INTEGER NOT NULL);
            INSERT OR IGNORE INTO audit_budget VALUES(1,0);
            CREATE TABLE IF NOT EXISTS audit_config (id INTEGER PRIMARY KEY CHECK(id=1), algorithm TEXT NOT NULL, key_id TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS audit_events (seq INTEGER PRIMARY KEY, at_ms INTEGER NOT NULL,
                kind TEXT NOT NULL, detail TEXT NOT NULL, previous TEXT NOT NULL, digest TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_events BEGIN SELECT RAISE(ABORT,'audit records are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_events BEGIN SELECT RAISE(ABORT,'audit records are append-only'); END;
        """)
        row = db.execute("SELECT algorithm,key_id FROM audit_config WHERE id=1").fetchone()
        if row is None:
            existing = db.execute("SELECT value FROM scheduler_meta WHERE key='jobs'").fetchone()[0]
            if existing:
                raise ValueError("existing unaudited jobs need explicit offline migration; cannot silently authenticate history")
            with db:
                db.execute("INSERT INTO audit_config VALUES(1,?,?)", (self.algorithm, self.key_id))
        elif tuple(row) != (self.algorithm, self.key_id):
            raise ValueError("audit key/algorithm mismatch or downgrade attempt")
        self.verify()

    def _digest(self, seq, at, kind, detail, previous):
        value = json.dumps(["hyperl-audit/1", self.algorithm, seq, at, kind, detail, previous], separators=(",", ":"), ensure_ascii=True).encode()
        return hmac.new(self.key, value, hashlib.sha256).hexdigest() if self.key is not None else hashlib.sha256(value).hexdigest()

    def append(self, at_ms, kind, detail):
        if not self.db.in_transaction:
            raise RuntimeError("audit record must share the authoritative state transaction")
        text = json.dumps(detail, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if not isinstance(kind, str) or not 1 <= len(kind) <= 64 or len(text.encode()) > 256 * 1024:
            raise ValueError("bounded audit record required")
        used = self.db.execute("SELECT bytes FROM audit_budget WHERE id=1").fetchone()[0]
        cost = len(text.encode()) + len(kind) + 256
        if used + cost > self.max_bytes:
            raise ValueError("audit storage quota exhausted; archive/reconcile externally before new authority changes")
        self.db.execute("UPDATE audit_budget SET bytes=bytes+? WHERE id=1", (cost,))
        row = self.db.execute("SELECT seq,digest FROM audit_events ORDER BY seq DESC LIMIT 1").fetchone()
        seq, previous = (row[0] + 1, row[1]) if row else (1, "0" * 64)
        digest = self._digest(seq, at_ms, kind, text, previous)
        self.db.execute("INSERT INTO audit_events VALUES(?,?,?,?,?,?)", (seq, at_ms, kind, text, previous, digest))

    def head(self):
        row = self.db.execute("SELECT seq,digest FROM audit_events ORDER BY seq DESC LIMIT 1").fetchone()
        return dict(format="hyperl-audit-head/1", sequence=row[0] if row else 0, digest=row[1] if row else "0" * 64,
                    algorithm=self.algorithm, key_id=self.key_id)

    def verify(self, expected_head=None):
        sequence = 0; previous = "0" * 64
        for seq, at, kind, detail, prior, digest in self.db.execute("SELECT seq,at_ms,kind,detail,previous,digest FROM audit_events ORDER BY seq"):
            if seq != sequence + 1 or prior != previous or not hmac.compare_digest(digest, self._digest(seq, at, kind, detail, prior)):
                raise ValueError("audit chain authentication/order failed; authority review required")
            sequence = seq; previous = digest
        head = dict(format="hyperl-audit-head/1", sequence=sequence, digest=previous, algorithm=self.algorithm, key_id=self.key_id)
        if expected_head is not None and head != expected_head:
            raise ValueError("external audit checkpoint mismatch/truncation")
        return head

    def page(self, after=0, limit=100):
        if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("bounded audit page required")
        return [dict(seq=r[0], at_ms=r[1], kind=r[2], detail=json.loads(r[3]), previous=r[4], digest=r[5])
                for r in self.db.execute("SELECT seq,at_ms,kind,detail,previous,digest FROM audit_events WHERE seq>? ORDER BY seq LIMIT ?", (after, limit))]
