# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
# Copyright 2026 Sabbir Hassan Imon and respective contributors.
"""Disk-backed inventory, not a million-device execution engine or network listener.

Each database is one directory shard. Bounded execution cells are enrolled explicitly;
callers federate shards by stable node ID. SQLite uses its native indexed storage engine.
Endpoints never contain credentials. Heartbeat evidence is supplied by the owner agent.
"""
from dataclasses import dataclass
import hashlib
import ipaddress
import itertools
import re
import sqlite3
from urllib.parse import urlsplit

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


@dataclass(frozen=True)
class Device:
    id: str
    cell: str
    architecture: str
    endpoint: str
    instance: str
    epoch: int
    expires_ms: int


def _name(value):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError("invalid directory identity")
    return value


def shard_for(node_id, shards):
    _name(node_id)
    if not isinstance(shards, int) or isinstance(shards, bool) or not 1 <= shards <= 65536:
        raise ValueError("invalid shard count")
    return int.from_bytes(hashlib.sha256(node_id.encode()).digest()[:8], "big") % shards


def _validate(device, now_ms):
    for value in (device.id, device.cell, device.architecture, device.instance):
        _name(value)
    if (not isinstance(device.epoch, int) or isinstance(device.epoch, bool) or not 1 <= device.epoch < 2**63
            or not now_ms < device.expires_ms <= now_ms + 300_000):
        raise ValueError("invalid epoch/lease; maximum heartbeat lease is five minutes")
    if len(device.endpoint) > 2048 or any(ord(c) <= 32 for c in device.endpoint):
        raise ValueError("invalid endpoint")
    uri = urlsplit(device.endpoint)
    if not uri.hostname or uri.username or uri.password or uri.path not in ("", "/") or uri.query or uri.fragment:
        raise ValueError("endpoint must be an origin without credentials")
    if uri.port is not None and not 1 <= uri.port <= 65535:
        raise ValueError("invalid endpoint port")
    if uri.scheme != "https":
        try:
            loopback = ipaddress.ip_address(uri.hostname).is_loopback
        except ValueError:
            loopback = False
        if uri.scheme != "http" or not loopback or uri.hostname not in ("127.0.0.1", "::1"):
            raise ValueError("remote devices require HTTPS")


class DeviceDirectory:
    """Persistent leases, epoch fences, named pools, indexed keyset pagination.

    This stores inventory only. A live cell selects at most its executor's bounded
    worker limit. A stale/restarted device is never implicitly reauthorized.
    Use a private, owner-selected database path; no credentials are stored here.
    """
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS devices (
                id TEXT PRIMARY KEY, cell TEXT NOT NULL, architecture TEXT NOT NULL,
                endpoint TEXT NOT NULL, instance TEXT NOT NULL, epoch INTEGER NOT NULL,
                expires_ms INTEGER NOT NULL, revoked INTEGER NOT NULL DEFAULT 0);
            CREATE INDEX IF NOT EXISTS devices_cell ON devices(cell,id);
            CREATE INDEX IF NOT EXISTS devices_expiry ON devices(expires_ms,id);
            CREATE TABLE IF NOT EXISTS pools (pool TEXT NOT NULL, cell TEXT NOT NULL,
                PRIMARY KEY(pool,cell));
        """)

    def close(self):
        self.db.close()

    def enroll(self, devices, now_ms):
        """One bounded atomic batch. Higher epoch is an explicit owner re-enrollment.

        Ordinary heartbeats cannot replace endpoint/architecture/instance or undo a
        revocation. Instance replacement requires a strictly larger epoch.
        """
        batch = list(itertools.islice(devices, 1001))
        if not 1 <= len(batch) <= 1000 or len({d.id for d in batch}) != len(batch):
            raise ValueError("enrollment batch must contain 1..1000 unique devices")
        for device in batch:
            _validate(device, now_ms)
        with self.db:
            for d in batch:
                old = self.db.execute("SELECT cell,architecture,endpoint,instance,epoch,revoked FROM devices WHERE id=?", (d.id,)).fetchone()
                if old and (d.epoch < old[4] or d.epoch == old[4] and (old[5] or tuple((d.cell, d.architecture, d.endpoint, d.instance)) != old[:4])):
                    raise ValueError("stale epoch, revoked identity or changed heartbeat evidence")
                self.db.execute("""INSERT INTO devices VALUES(?,?,?,?,?,?,?,0)
                    ON CONFLICT(id) DO UPDATE SET cell=excluded.cell,architecture=excluded.architecture,
                    endpoint=excluded.endpoint,instance=excluded.instance,epoch=excluded.epoch,
                    expires_ms=excluded.expires_ms,revoked=0""",
                    (d.id, d.cell, d.architecture, d.endpoint, d.instance, d.epoch, d.expires_ms))

    def revoke(self, node_id, epoch):
        with self.db:
            return self.db.execute("UPDATE devices SET revoked=1 WHERE id=? AND epoch=?", (_name(node_id), epoch)).rowcount == 1

    def set_pool(self, pool, cells):
        _name(pool)
        cells = list(itertools.islice(cells, 1001))
        if not 1 <= len(cells) <= 1000 or len(set(cells)) != len(cells):
            raise ValueError("one pool contains 1..1000 directory cells; nest hierarchy outside this shard")
        for cell in cells:
            _name(cell)
        with self.db:
            self.db.execute("DELETE FROM pools WHERE pool=?", (pool,))
            self.db.executemany("INSERT INTO pools VALUES(?,?)", ((pool, c) for c in cells))

    def page(self, now_ms, *, cell=None, pool=None, after="", limit=1000):
        """Live leases only. Cursor is last returned ID; no OFFSET or full fleet load.

        Pages are current observations, not a transaction snapshot or dispatch approval.
        Revalidate each chosen identity/epoch/lease immediately before execution.
        """
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 1000 or cell is not None and pool is not None:
            raise ValueError("invalid directory page")
        if after:
            _name(after)
        params = [after, now_ms]
        where = "d.id>? AND d.expires_ms>? AND d.revoked=0"
        if cell is not None:
            where += " AND d.cell=?"
            params.append(_name(cell))
        if pool is not None:
            where += " AND EXISTS(SELECT 1 FROM pools p WHERE p.cell=d.cell AND p.pool=?)"
            params.append(_name(pool))
        params.append(limit)
        rows = self.db.execute("SELECT d.id,d.cell,d.architecture,d.endpoint,d.instance,d.epoch,d.expires_ms FROM devices d WHERE " + where + " ORDER BY d.id LIMIT ?", params)
        return [Device(*row) for row in rows]

    def resolve(self, node_id, epoch, now_ms):
        row = self.db.execute("SELECT id,cell,architecture,endpoint,instance,epoch,expires_ms FROM devices WHERE id=? AND epoch=? AND expires_ms>? AND revoked=0", (_name(node_id), epoch, now_ms)).fetchone()
        if row is None:
            raise ValueError("device unavailable, expired, revoked or fenced")
        return Device(*row)
