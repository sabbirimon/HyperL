#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
"""Explicit pretrained GGUF inference, native layer/KV sharding, mTLS and live pull jobs.

Only caller-selected local binaries/models; no model downloads, paid service or GPU
fallback. Two CPU worker processes on one host are not two physical architectures.
"""
import argparse
import asyncio
import dataclasses
import hashlib
import json
from pathlib import Path
import secrets
import sys
import tempfile
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from hyperl.llm import LlamaSettings, LlamaSession, RpcWorker, file_digest, _port
from hyperl.connections import RpcTunnel, server_tls, client_tls
from hyperl.scheduler import Cell, FleetScheduler
from hyperl.fleet_service import FleetService, FleetClient, LlmAgent
from tls_fixtures import certificates


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def model_check(baseline, result):
    require(result["tokens"] == baseline["tokens"] and bool(result["tokens"]), "split generation differs from greedy local reference")
    require(result["text"] == baseline["text"], "split detokenization differs")
    layout = result["native"]
    require({"RPC0", "RPC1"} <= set(layout["assigned_layers"].values()), "both workers must own actual model layers")
    require(len(layout["kv_buffers_mib"]) >= 2 and all(v > 0 for v in layout["kv_buffers_mib"].values()), "both workers need actual KV allocations")


async def qualify(args):
    server = str(args.server.resolve()); worker = str(args.worker.resolve()); model = str(args.model.resolve())
    require(file_digest(model) == args.model_sha256, "model digest mismatch before execution")
    settings = LlamaSettings(server, file_digest(server), model, args.model_sha256, context=512, request_seconds=30)
    prompt = "Once upon a time"
    local = await asyncio.to_thread(LlamaSession, settings)
    try:
        baseline = await asyncio.to_thread(local.complete, prompt, max_tokens=16)
    finally:
        await asyncio.to_thread(local.close)
    workers = []; tls_servers = []; tunnels = []; split = None; fleet = None; scheduler = None; agent = None
    with tempfile.TemporaryDirectory(prefix="hyperl-llm-proof-") as temporary:
        try:
            for _ in range(2):
                workers.append(await asyncio.to_thread(RpcWorker, worker, file_digest(worker), port=_port(), device="CPU"))
            direct_settings = dataclasses.replace(settings, backend="RPC", rpc=tuple(w.endpoint for w in workers), proportions=(1, 1))
            split = await asyncio.to_thread(LlamaSession, direct_settings)
            direct = await asyncio.to_thread(split.complete, prompt, max_tokens=16)
            model_check(baseline, direct)
            require(all(w.evidence()["graph_calls"] > 0 for w in workers), "both CPU workers must execute native graphs")
            await asyncio.to_thread(split.close); split = None
            certs = await asyncio.to_thread(certificates, Path(temporary))
            for w in workers:
                remote = await RpcTunnel(listen_tls=server_tls(*certs["server"]), target_port=int(w.endpoint.split(":")[1]), allowed_peers=("cell:cell-a",)).start()
                tls_servers.append(remote)
                local_route = await RpcTunnel(target_host="127.0.0.1", target_port=remote.port,
                    target_tls=client_tls(*certs["cell:cell-a"]), server_hostname="127.0.0.1").start()
                tunnels.append(local_route)
            secure_settings = dataclasses.replace(direct_settings, rpc=tuple(f"127.0.0.1:{route.port}" for route in tunnels))
            split = await asyncio.to_thread(LlamaSession, secure_settings)
            secure = await asyncio.to_thread(split.complete, prompt, max_tokens=16)
            model_check(baseline, secure)
            await asyncio.to_thread(split.close); split = None
            require(all(t.forward_bytes > 0 and t.return_bytes > 0 for t in tunnels), "each TLS route must carry real model traffic")
            scheduler = FleetScheduler(Path(temporary) / "jobs.sqlite", audit_key=secrets.token_bytes(32))
            cell = Cell("cell-a", "proof-instance", 1, ("combined",), ("pretrained",))
            scheduler.register_profile("pretrained", settings.model_sha256, "llama.cpp")
            scheduler.register(cell)
            fleet = await FleetService(scheduler, server_tls(*certs["server"])).start()
            origin = f"https://127.0.0.1:{fleet.port}"
            tenant = FleetClient(origin, client_tls(*certs["tenant:tenant-a"]))
            client = FleetClient(origin, client_tls(*certs["cell:cell-a"]))
            agent = LlmAgent(client, cell, {"pretrained": secure_settings})
            ids = await asyncio.to_thread(tenant.call, "/v1/jobs/submit", dict(jobs=[dict(key=f"job-{i}", pool="combined", profile="pretrained", payload=dict(prompt=prompt, max_tokens=16)) for i in range(2)]))
            for identifier in ids:
                require(await asyncio.to_thread(agent.run_once), "live agent did not receive the queued model job")
                status = await asyncio.to_thread(tenant.call, "/v1/jobs/get", dict(id=identifier))
                require(status["state"] == "SUCCEEDED", "live native job failed")
                model_check(baseline, status["result"])
            audit_head = scheduler.audit.verify()
            require(audit_head["algorithm"] == "HMAC-SHA256/1", "live job evidence needs authenticated audit")
            await asyncio.to_thread(agent.close); agent = None
            # A worker dies after inference dispatch: the result must not publish.
            split = await asyncio.to_thread(LlamaSession, dataclasses.replace(secure_settings, request_seconds=10))
            before = sum(w.evidence()["graph_calls"] + w.evidence()["recompute_calls"] for w in workers)
            task = asyncio.create_task(asyncio.to_thread(split.complete, prompt, max_tokens=256))
            deadline = time.monotonic() + 5
            while sum(w.evidence()["graph_calls"] + w.evidence()["recompute_calls"] for w in workers) <= before and not task.done():
                require(time.monotonic() < deadline, "fault injection did not observe native dispatch")
                await asyncio.sleep(.005)
            await asyncio.to_thread(workers[1].close)
            failed = False
            try:
                await asyncio.wait_for(task, 15)
            except (OSError, RuntimeError, TimeoutError, ValueError):
                failed = True
            require(failed, "worker death incorrectly produced an accepted result")
            evidence = dict(format="hyperl-llm-qualification/1", source_revision=settings.upstream_revision,
                model_file=args.model.name, model_sha256=args.model_sha256, model_bytes=args.model.stat().st_size,
                local_tokens=baseline["tokens"], greedy_token_match=True, direct_rpc=True, mutual_tls_rpc=True,
                model_layer_assignment=secure["native"]["assigned_layers"], kv_buffers_mib=secure["native"]["kv_buffers_mib"],
                model_buffers_mib=secure["native"]["model_buffers_mib"], worker_graphs=[w.evidence() for w in workers],
                live_authenticated_model_jobs=len(ids), audit_head=audit_head, worker_loss_rejected=True,
                startup_seconds=secure["startup_seconds"], request_seconds=secure["request_seconds"],
                physical_hosts=1, cpu_worker_processes=2, physical_tpu_tested=False, physical_gpu_tested=False,
                million_live_devices_tested=False, speedup_claimed=False)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(evidence, indent=2) + "\n")
            print(json.dumps(evidence, indent=2))
        finally:
            if agent:
                await asyncio.to_thread(agent.close)
            if split:
                await asyncio.to_thread(split.close)
            if fleet:
                await fleet.close()
            if scheduler:
                scheduler.close()
            for route in tunnels:
                await route.close()
            for route in tls_servers:
                await route.close()
            for w in workers:
                await asyncio.to_thread(w.close)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output", type=Path)
    asyncio.run(qualify(parser.parse_args()))
