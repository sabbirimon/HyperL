# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
# Copyright 2026 Sabbir Hassan Imon and respective contributors.
"""Durable pull scheduling for bounded model-execution cells.

SQLite is the authoritative ledger, not a RAM queue/cache. Shard databases by
stable identity; agents talk to their configured shard. No automatic uncertain
replay, guessed TPU capacity, global million-socket loop, or exactly-once effects.
"""
from dataclasses import dataclass
import hashlib
import itertools
import json
import re
import sqlite3
import time
import uuid
from .audit import AuditTrail

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_ACTIVE = ("LEASED", "RUNNING", "CANCEL_REQUESTED")
_TERMINAL = ("SUCCEEDED", "FAILED", "CANCELLED", "UNCERTAIN")


def _name(value):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError("invalid scheduler identity")
    return value


def _json(value, maximum=65536):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(data.encode()) > maximum:
        raise ValueError("scheduler payload exceeds byte quota")
    return data


@dataclass(frozen=True)
class Cell:
    id: str
    instance: str
    epoch: int
    pools: tuple
    profiles: tuple
    slots: int = 1

    def validate(self):
        _name(self.id); _name(self.instance)
        if type(self.epoch) is not int or not 1 <= self.epoch < 2**63 or type(self.slots) is not int or not 1 <= self.slots <= 16:
            raise ValueError("invalid cell epoch/slots")
        for values in (self.pools, self.profiles):
            if not 1 <= len(values) <= 16 or len(set(values)) != len(values):
                raise ValueError("cell needs bounded unique pools/profiles")
            for value in values:
                _name(value)


class FleetScheduler:
    def __init__(self, path, *, max_jobs=10_000_000, max_pending_per_tenant=1000000, audit_key=None):
        if type(max_jobs) is not int or not 1 <= max_jobs <= 100_000_000 or type(max_pending_per_tenant) is not int or not 1 <= max_pending_per_tenant <= max_jobs:
            raise ValueError("invalid scheduler quotas")
        self.max_jobs = max_jobs; self.tenant_quota = max_pending_per_tenant; self.require_profiles = False
        self.db = sqlite3.connect(path, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS scheduler_meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
            INSERT OR IGNORE INTO scheduler_meta VALUES('clock',0),('jobs',0),('dispatch',0);
            CREATE TABLE IF NOT EXISTS model_profiles (id TEXT PRIMARY KEY, digest TEXT NOT NULL, adapter TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS cells (id TEXT PRIMARY KEY, instance TEXT NOT NULL,
                epoch INTEGER NOT NULL, pools TEXT NOT NULL, profiles TEXT NOT NULL,
                slots INTEGER NOT NULL, expires INTEGER NOT NULL, revoked INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS jobs (seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT UNIQUE NOT NULL, tenant TEXT NOT NULL, key TEXT NOT NULL,
                pool TEXT NOT NULL, profile TEXT NOT NULL, payload TEXT NOT NULL,
                digest TEXT NOT NULL, state TEXT NOT NULL, created INTEGER NOT NULL,
                cell TEXT, cell_instance TEXT, cell_epoch INTEGER, lease TEXT, expires INTEGER,
                result TEXT, result_digest TEXT, UNIQUE(tenant,key));
            CREATE INDEX IF NOT EXISTS jobs_ready ON jobs(pool,profile,tenant,state,seq);
            CREATE INDEX IF NOT EXISTS jobs_expiry ON jobs(state,expires,seq);
            CREATE INDEX IF NOT EXISTS jobs_cell ON jobs(cell,state,seq);
            CREATE INDEX IF NOT EXISTS jobs_tenant ON jobs(tenant,state,seq);
            CREATE TABLE IF NOT EXISTS queues (pool TEXT NOT NULL, profile TEXT NOT NULL,
                tenant TEXT NOT NULL, pending INTEGER NOT NULL, dispatched INTEGER NOT NULL,
                PRIMARY KEY(pool,profile,tenant));
            CREATE INDEX IF NOT EXISTS queues_ready ON queues(pool,profile,dispatched) WHERE pending>0;
        """)

        try:
            self.audit = AuditTrail(self.db, audit_key)
        except BaseException:
            self.db.close(); raise

    def close(self):
        self.db.close()

    def _now(self, now):
        if not self.db.in_transaction:
            self.db.execute("BEGIN IMMEDIATE")
        value = int(time.time() * 1000) if now is None else now
        if type(value) is not int or not 0 <= value < 2**63 - 600000:
            raise ValueError("invalid scheduler clock")
        previous = self.db.execute("SELECT value FROM scheduler_meta WHERE key='clock'").fetchone()[0]
        if value < previous:
            raise ValueError("scheduler clock moved backwards; authority must be reconciled before extending leases")
        self.db.execute("UPDATE scheduler_meta SET value=? WHERE key='clock'", (value,))
        return value

    def register_profile(self, name, digest, adapter, *, now=None):
        _name(name)
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest) or adapter not in {"llama.cpp", "jax-llama-f32"}:
            raise ValueError("exact immutable model/adapter registration required")
        with self.db:
            clock = self._now(now)
            old = self.db.execute("SELECT digest,adapter,revoked FROM model_profiles WHERE id=?", (name,)).fetchone()
            if old and tuple(old) != (digest, adapter, 0):
                raise ValueError("profile changed/revoked; enroll a new versioned profile ID")
            self.db.execute("INSERT OR IGNORE INTO model_profiles VALUES(?,?,?,0)", (name, digest, adapter))
            self.audit.append(clock, "MODEL_ENROLL", dict(profile=name, sha256=digest, adapter=adapter))

    def revoke_profile(self, name, *, now=None):
        with self.db:
            clock = self._now(now)
            self.db.execute("UPDATE model_profiles SET revoked=1 WHERE id=?", (_name(name),))
            self.db.execute("UPDATE jobs SET state='UNCERTAIN' WHERE profile=? AND state IN ('LEASED','RUNNING','CANCEL_REQUESTED')", (name,))
            self.audit.append(clock, "MODEL_REVOKE", dict(profile=name))

    def _profile(self, name):
        row = self.db.execute("SELECT digest,adapter,revoked FROM model_profiles WHERE id=?", (name,)).fetchone()
        if row is None:
            if self.require_profiles:
                raise ValueError("model profile is not owner registered")
            return None
        if row["revoked"]:
            raise ValueError("model profile revoked")
        return row

    def register(self, cell, *, now=None, ttl_ms=30000):
        cell.validate()
        if type(ttl_ms) is not int or not 1000 <= ttl_ms <= 300000:
            raise ValueError("cell heartbeat TTL outside 1s..5m")
        with self.db:
            clock = self._now(now)
            old = self.db.execute("SELECT * FROM cells WHERE id=?", (cell.id,)).fetchone()
            if old and (cell.epoch < old["epoch"] or cell.epoch == old["epoch"] and (old["revoked"] or cell.instance != old["instance"]
                    or _json(cell.pools) != old["pools"] or _json(cell.profiles) != old["profiles"] or cell.slots != old["slots"])):
                raise ValueError("cell restart/revocation requires a higher owner-enrolled epoch")
            if old and cell.epoch > old["epoch"]:
                self.db.execute("UPDATE jobs SET state='UNCERTAIN' WHERE cell=? AND state IN ('LEASED','RUNNING','CANCEL_REQUESTED')", (cell.id,))
            self.db.execute("""INSERT INTO cells VALUES(?,?,?,?,?,?,?,0) ON CONFLICT(id) DO UPDATE SET
                instance=excluded.instance,epoch=excluded.epoch,pools=excluded.pools,profiles=excluded.profiles,
                slots=excluded.slots,expires=excluded.expires,revoked=0""",
                (cell.id, cell.instance, cell.epoch, _json(cell.pools), _json(cell.profiles), cell.slots, clock + ttl_ms))
            self.audit.append(clock, "CELL_ENROLL", dict(id=cell.id, instance=cell.instance, epoch=cell.epoch))

    def heartbeat(self, cell, *, now=None, ttl_ms=30000):
        # Agent heartbeats can renew existing evidence only; they cannot enroll/re-authorize.
        cell.validate()
        with self.db:
            clock = self._now(now)
            current = self._cell(cell.id, cell.instance, cell.epoch, clock)
            if current["pools"] != _json(cell.pools) or current["profiles"] != _json(cell.profiles) or current["slots"] != cell.slots:
                raise ValueError("agent capability change needs owner enrollment")
            if type(ttl_ms) is not int or not 1000 <= ttl_ms <= 300000:
                raise ValueError("heartbeat TTL outside bounds")
            self.db.execute("UPDATE cells SET expires=? WHERE id=?", (clock + ttl_ms, cell.id))
            self.audit.append(clock, "CELL_RENEW", dict(id=cell.id, epoch=cell.epoch, expires=clock + ttl_ms))

    def revoke(self, cell_id, *, now=None):
        with self.db:
            clock = self._now(now)
            self.db.execute("UPDATE cells SET revoked=1 WHERE id=?", (_name(cell_id),))
            self.db.execute("UPDATE jobs SET state='UNCERTAIN' WHERE cell=? AND state IN ('LEASED','RUNNING','CANCEL_REQUESTED')", (cell_id,))
            self.audit.append(clock, "CELL_REVOKE", dict(id=cell_id))

    def submit(self, tenant, requests, *, now=None):
        _name(tenant)
        batch = list(itertools.islice(requests, 1001))
        if not 1 <= len(batch) <= 1000:
            raise ValueError("submit 1..1000 bounded job descriptions")
        prepared = []
        for item in batch:
            key = _name(item["key"]); pool = _name(item["pool"]); profile = _name(item["profile"])
            self._profile(profile)
            payload = _json(item["payload"])
            digest = hashlib.sha256(_json(dict(pool=pool, profile=profile, payload=json.loads(payload))).encode()).hexdigest()
            prepared.append((key, pool, profile, payload, digest))
        result = []
        with self.db:
            clock = self._now(now)
            pending = self.db.execute("SELECT coalesce(sum(pending),0) FROM queues WHERE tenant=?", (tenant,)).fetchone()[0]
            count = self.db.execute("SELECT value FROM scheduler_meta WHERE key='jobs'").fetchone()[0]
            for key, pool, profile, payload, digest in prepared:
                self._profile(profile)
                old = self.db.execute("SELECT id,digest FROM jobs WHERE tenant=? AND key=?", (tenant, key)).fetchone()
                if old:
                    if old["digest"] != digest:
                        raise ValueError("idempotency key reused with different work")
                    result.append(old["id"]); continue
                if count >= self.max_jobs or pending >= self.tenant_quota:
                    raise ValueError("scheduler/tenant durable queue quota reached")
                identifier = str(uuid.uuid4())
                self.db.execute("INSERT INTO jobs(id,tenant,key,pool,profile,payload,digest,state,created) VALUES(?,?,?,?,?,?,?,'READY',?)",
                    (identifier, tenant, key, pool, profile, payload, digest, clock))
                self.db.execute("""INSERT INTO queues VALUES(?,?,?,1,0) ON CONFLICT(pool,profile,tenant)
                    DO UPDATE SET pending=pending+1""", (pool, profile, tenant))
                result.append(identifier); count += 1; pending += 1
            self.db.execute("UPDATE scheduler_meta SET value=? WHERE key='jobs'", (count,))
            self.audit.append(clock, "SUBMIT", dict(tenant=tenant, jobs=result, inputs=[item[4] for item in prepared]))
        return result

    def _cell(self, cell_id, instance, epoch, now):
        row = self.db.execute("SELECT * FROM cells WHERE id=? AND instance=? AND epoch=? AND expires>? AND revoked=0", (cell_id, instance, epoch, now)).fetchone()
        if row is None:
            raise ValueError("cell lease/epoch revoked, expired or changed")
        return row

    def claim(self, cell_id, instance, epoch, *, now=None, lease_ms=30000):
        if type(lease_ms) is not int or not 1000 <= lease_ms <= 300000:
            raise ValueError("job lease outside bounds")
        with self.db:
            clock = self._now(now); cell = self._cell(cell_id, instance, epoch, clock)
            active = self.db.execute("SELECT count(*) FROM jobs WHERE cell=? AND state IN ('LEASED','RUNNING','CANCEL_REQUESTED')", (cell_id,)).fetchone()[0]
            if active >= cell["slots"]:
                return None
            pools = json.loads(cell["pools"]); profiles = json.loads(cell["profiles"])
            query = "SELECT * FROM queues WHERE pending>0 AND pool IN (" + ",".join("?" for _ in pools) + ") AND profile IN (" + ",".join("?" for _ in profiles) + ") ORDER BY dispatched,pool,profile,tenant LIMIT 1"
            queue = self.db.execute(query, (*pools, *profiles)).fetchone()
            if queue is None:
                return None
            row = self.db.execute("SELECT * FROM jobs WHERE pool=? AND profile=? AND tenant=? AND state='READY' ORDER BY seq LIMIT 1", (queue["pool"], queue["profile"], queue["tenant"])).fetchone()
            if row is None:
                raise RuntimeError("authoritative queue index mismatch")
            model = self._profile(row["profile"])
            token = str(uuid.uuid4())
            expires = min(clock + lease_ms, cell["expires"])
            self.db.execute("UPDATE jobs SET state='LEASED',cell=?,cell_instance=?,cell_epoch=?,lease=?,expires=? WHERE id=? AND state='READY'",
                            (cell_id, instance, epoch, token, expires, row["id"]))
            self.db.execute("UPDATE scheduler_meta SET value=value+1 WHERE key='dispatch'")
            dispatched = self.db.execute("SELECT value FROM scheduler_meta WHERE key='dispatch'").fetchone()[0]
            self.db.execute("UPDATE queues SET pending=pending-1,dispatched=? WHERE pool=? AND profile=? AND tenant=?", (dispatched, queue["pool"], queue["profile"], queue["tenant"]))
            self.audit.append(clock, "CLAIM", dict(job=row["id"], cell=cell_id, epoch=epoch, input_sha256=row["digest"]))
            return dict(id=row["id"], tenant=row["tenant"], profile=row["profile"], payload=json.loads(row["payload"]),
                        input_sha256=row["digest"], artifact_sha256=model["digest"] if model else None, adapter=model["adapter"] if model else None, lease=token, expires_ms=expires, cell=cell_id, instance=instance, epoch=epoch)

    def _owned(self, job_id, lease, cell_id, instance, epoch, now):
        self._cell(cell_id, instance, epoch, now)
        row = self.db.execute("SELECT * FROM jobs WHERE id=? AND lease=? AND cell=? AND cell_instance=? AND cell_epoch=?", (job_id, lease, cell_id, instance, epoch)).fetchone()
        if row is None or row["state"] not in _ACTIVE or row["expires"] <= now:
            raise ValueError("stale, expired or mismatched job authority; no result publication")
        self._profile(row["profile"])
        return row

    def start(self, receipt, *, now=None):
        with self.db:
            clock = self._now(now)
            row = self._owned(receipt["id"], receipt["lease"], receipt["cell"], receipt["instance"], receipt["epoch"], clock)
            if row["state"] != "LEASED":
                raise ValueError("job cannot start twice or after cancellation")
            self.db.execute("UPDATE jobs SET state='RUNNING' WHERE id=?", (row["id"],))
            self.audit.append(clock, "START", dict(job=row["id"], cell=row["cell"], epoch=row["cell_epoch"]))

    def renew(self, receipt, *, now=None, lease_ms=30000):
        if type(lease_ms) is not int or not 1000 <= lease_ms <= 300000:
            raise ValueError("job lease outside bounds")
        with self.db:
            clock = self._now(now)
            row = self._owned(receipt["id"], receipt["lease"], receipt["cell"], receipt["instance"], receipt["epoch"], clock)
            cell = self._cell(receipt["cell"], receipt["instance"], receipt["epoch"], clock)
            expiry = min(clock + lease_ms, cell["expires"])
            self.db.execute("UPDATE jobs SET expires=? WHERE id=?", (expiry, row["id"]))
            self.audit.append(clock, "LEASE_RENEW", dict(job=row["id"], state=row["state"], expires=expiry))
            return dict(state=row["state"], expires_ms=expiry)

    def finish(self, receipt, result, *, state="SUCCEEDED", now=None):
        if state not in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            raise ValueError("explicit terminal worker acknowledgment required")
        payload = _json(result, 1024 * 1024); digest = hashlib.sha256(payload.encode()).hexdigest()
        with self.db:
            clock = self._now(now)
            old = self.db.execute("SELECT state,result_digest,lease FROM jobs WHERE id=?", (receipt["id"],)).fetchone()
            if old and old["state"] in ("SUCCEEDED", "FAILED", "CANCELLED") and old["lease"] == receipt["lease"] and old["state"] == state and old["result_digest"] == digest:
                return digest  # idempotent acknowledgment, not repeated execution
            row = self._owned(receipt["id"], receipt["lease"], receipt["cell"], receipt["instance"], receipt["epoch"], clock)
            if row["state"] == "CANCEL_REQUESTED" and state != "CANCELLED":
                raise ValueError("cancellation revoked result publication")
            if row["state"] == "LEASED" and state == "SUCCEEDED":
                raise ValueError("unstarted job cannot publish success")
            model = self._profile(row["profile"])
            if state == "SUCCEEDED" and model is not None:
                field = "model_sha256" if model["adapter"] == "llama.cpp" else "checkpoint_manifest_sha256"
                if not isinstance(result, dict) or result.get(field) != model["digest"]:
                    raise ValueError("worker result model identity differs from approved profile")
            self.db.execute("UPDATE jobs SET state=?,result=?,result_digest=? WHERE id=?", (state, payload, digest, row["id"]))
            self.audit.append(clock, state, dict(job=row["id"], cell=row["cell"], result_sha256=digest))
            return digest

    def cancel(self, tenant, job_id, *, now=None):
        with self.db:
            clock = self._now(now)
            row = self.db.execute("SELECT * FROM jobs WHERE id=? AND tenant=?", (job_id, _name(tenant))).fetchone()
            if row is None:
                raise ValueError("unknown tenant/job")
            if row["state"] == "READY":
                self.db.execute("UPDATE jobs SET state='CANCELLED' WHERE id=?", (job_id,))
                self.db.execute("UPDATE queues SET pending=pending-1 WHERE pool=? AND profile=? AND tenant=?", (row["pool"], row["profile"], tenant))
            elif row["state"] in _ACTIVE:
                self.db.execute("UPDATE jobs SET state='CANCEL_REQUESTED' WHERE id=?", (job_id,))

            self.audit.append(clock, "CANCEL_REQUEST", dict(job=job_id, tenant=tenant))

    def expire(self, *, now=None, limit=1000):
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("bounded reconciliation page required")
        with self.db:
            clock = self._now(now)
            # A lost agent is uncertain; do not blindly replay stateful model work.
            rows = self.db.execute("SELECT seq FROM jobs WHERE state IN ('LEASED','RUNNING','CANCEL_REQUESTED') AND expires<=? ORDER BY expires,seq LIMIT ?", (clock, limit)).fetchall()
            self.db.executemany("UPDATE jobs SET state='UNCERTAIN' WHERE seq=?", ((r[0],) for r in rows))
            if rows:
                self.audit.append(clock, "EXPIRE_UNCERTAIN", dict(sequences=[row[0] for row in rows]))
            return len(rows)

    def job(self, tenant, identifier):
        row = self.db.execute("SELECT id,seq,profile,state,result,result_digest,cell,expires FROM jobs WHERE tenant=? AND id=?", (_name(tenant), identifier)).fetchone()
        if row is None:
            raise ValueError("unknown tenant/job")
        result = dict(row)
        if result["result"] is not None:
            result["result"] = json.loads(result["result"])
        return result

    def page(self, tenant, *, after=0, limit=100):
        if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("invalid job page")
        return [dict(row) for row in self.db.execute("SELECT id,seq,profile,state,cell FROM jobs WHERE tenant=? AND seq>? ORDER BY seq LIMIT ?", (_name(tenant), after, limit))]

    def record_denial(self, identity, route):
        with self.db:
            clock = self._now(None)
            self.audit.append(clock, "DENY", dict(identity=str(identity)[:128], route=str(route)[:128]))
