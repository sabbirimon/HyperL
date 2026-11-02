# Integrity, recovery and high-assurance engineering baseline

2026-11-02. Owner-selected scope: a general NIST control mapping for HyperL,
including possible defence-oriented deployments. This is an engineering mapping
and gap record, **not a certification, authorization to operate, classified-system
approval, military standard conformity claim or independent assessment**.

NIST SP 800-53 Rev. 5 is the reference control catalog. Organizations must select
and assess their actual system controls; a library test cannot establish the whole
operating environment's compliance. [Official NIST catalog](https://csrc.nist.gov/pubs/sp/800/53/r5/upd1/final).
Using standard algorithms also does not establish FIPS 140-3 module validation.
[Official FIPS 140-3 publication](https://csrc.nist.gov/pubs/fips/140-3/final) and
[CMVP guidance](https://csrc.nist.gov/projects/cryptographic-module-validation-program/faqs).

## Storage error correction

`native/erasure.c` adds a separate ABI-1 systematic Cauchy Reed-Solomon codec over
GF(256), polynomial `0x11d`. It supports 1–16 data fragments and 1–4 parity
fragments, with 1 byte–4 MiB per fragment. Original data fragments are unchanged.
Any at-most-`m` missing fragments in a `k+m` stripe can be reconstructed from `k`
verified survivors. Native cancellation polls every 4 KiB; partial outputs must be
discarded. Bounds, pointer ranges and input/output aliasing are checked before work.
The existing CPU numerical ABI is unchanged.

The codec performs **erasure correction**: trusted SHA-256 hashes first identify
missing/corrupt fragments; reconstructed data must then match the trusted digest.
It does not guess which unverified fragments are wrong. Native tests compare parity
against an independent polynomial arithmetic oracle and exercise mixed data/parity
loss, insufficient survivors, alias rejection and cancellation. Python tests repair
actual model-store blobs, then require a complete original-artifact hash match.

Build the separate library with the ordinary CMake workflow. On macOS its example
name is `libhyperl_erasure.dylib`; Linux uses `.so`, Windows `.dll`.

```sh
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --config Release
ctest --test-dir build/native -C Release --output-on-failure
PYTHONPATH=python python3 -m hyperl.cluster model-protect \
  /owner/model-store ARTIFACT_UUID MAX_BYTES \
  --library /owner/lib/libhyperl_erasure.so --library-sha256 ACTUAL_LIBRARY_SHA256 \
  --data-shards 4 --parity-shards 2
PYTHONPATH=python python3 -m hyperl.cluster model-repair \
  /owner/model-store ARTIFACT_UUID MAX_BYTES \
  --library /owner/lib/libhyperl_erasure.so --library-sha256 ACTUAL_LIBRARY_SHA256 \
  --manifest-sha256 INDEPENDENTLY_TRUSTED_PROTECTION_DIGEST
```

Choose the real platform library and numeric quota. Protection returns an immutable
manifest digest. Keep that digest under separately trusted policy, such as an
externally signed artifact manifest. The repair operation checks the manifest before
reading fragments, restores only digest-matching content-addressed blobs and records
a repair event after full artifact verification. Existing model exports still
publish only after the original whole-artifact check.

Parity costs `m/k` extra full-stripe storage before filesystem/dedup overhead. Padding
is explicit for the final stripe. One damaged deduplicated blob can represent several
logical erasures, and failures above the stripe budget reject. Protection uses
existing chunks no larger than 4 MiB; larger imported chunk sizes require an explicit
rechunking migration. A cancelled or later-failing repair can leave independently
verified repaired chunks; no partly verified model export is accepted. Failed
protection can leave unreferenced reusable parity blobs, with no automatic deletion.

**Failure domains matter.** Parity in one directory protects some missing/bit-rotted
files; it cannot survive losing the entire disk or host. Copy verified stores and
protection manifests to independent failure domains and test restores. This is not
a distributed storage service, automatic cross-site repair, hardware RAM/VRAM ECC,
Byzantine consensus, secure erasure or a durability SLA. Independently assess native
memory safety and physical device error telemetry before critical deployment.

## Authenticated audit and fail-closed authority

The fleet's job-state and audit changes share one SQLite transaction. Records bind
sequence, time, event category, actor/job metadata, content digests and the preceding
record. The remote service requires a separately stored 32-byte HMAC key. Key and
algorithm mismatches reject opening the ledger; an existing unaudited job history
cannot be silently relabelled authenticated. Prompts, API keys, native log text,
model tensors and lease credentials are not audit details.

```sh
PYTHONPATH=python python3 -m hyperl.cluster audit-key-new /owner/keys/audit.key
PYTHONPATH=python python3 -m hyperl.cluster audit-verify /owner/fleet-service.json
PYTHONPATH=python python3 -m hyperl.cluster audit-verify /owner/fleet-service.json --checkpoint /owner/external-trusted-head.json
```

Key creation never overwrites. POSIX readers reject group/other-readable audit keys;
Windows ACLs and protected key provisioning remain operator tasks. The key stays
outside language/model JSON and logs. The configuration contains only its path.

Ordinary SQL updates/deletes of audit rows reject. HMAC verification detects changed
records; an independently preserved head detects suffix truncation. Keep heads in
an external protected archive and monitor freshness/continuity. A host administrator
who controls both the key and database remains inside the trust boundary. These
records do not provide WORM storage, public signatures or nonrepudiation. The default
1 GiB logical audit-record budget fails state changes closed when exhausted;
filesystem overhead and the job database need separate operator quotas. External
archival/rotation needs a verified migration procedure; nothing silently discards
old security events.

The direct Python scheduler can create explicitly labelled unkeyed laboratory
chains for isolated tests. Such ledgers cannot start the authenticated remote
service. They are not an assurance substitute for the keyed deployment path.

## Engineering mapping to NIST SP 800-53 Rev. 5

The following mapping is HyperL's interpretation, with deliberately bounded status.
It describes contributions to controls, not completed control implementations for
an organization. Test filenames are reproducible evidence, not certification.

| Control IDs | Implemented contribution | Status and remaining environment work |
|---|---|---|
| AC-3, AC-6 | CA-issued tenant/cell identities bind API routes; jobs select owner-installed profiles; incoming requests cannot choose executable/model paths | Partial: deployment IAM, OS sandboxing, admin separation and policy review required |
| IA-3, IA-5 | mTLS, verified hostnames, exact worker instances/epochs, private external keys | Partial: PKI issuance, revocation/rotation, hardware key custody and identity lifecycle required |
| AU-2, AU-9, AU-12 | Transactional HMAC event chain, bounded pages, external-head verification, ordinary append-only SQL guards and explicit audit quota | Partial: external archive, retention, alerting, clock authority and incident review required |
| CM-6, CM-7 | Explicit backend profiles, fixed native numerical settings, pinned source/dependencies, offline native loading, bounded allowed API routes | Partial: configuration approval, SBOM/transitive review, patch process and hardened host images required |
| SC-8, SC-13 | mTLS transport and standard SHA-256/HMAC; existing separately scoped HLM2/Ed25519 facilities remain | Partial: no CMVP-validated cryptographic configuration is claimed; organization-specific cryptographic policy required |
| SC-28 | Private selected artifact/checkpoint/key directories; existing HLM2 API can store encrypted data separately | Gap for automatic model/KV encryption: model-store and JAX checkpoint files are plaintext; provision qualified encrypted storage |
| SI-7, SI-10 | Exact artifact/chunk manifests, bounded parsing/frames, shape/dtype checks, trusted-erasure detection and checked reconstruction | Partial: signed provenance/trust lifecycle, hardware ECC telemetry, fuzzing and independent native-code review required |
| CP-9, CP-10 | Streaming immutable artifacts, parity repair, checked token-boundary KV checkpoints, durable jobs and restart/late-result fences | Partial: replicated backups, cross-site restoration, coordinated multi-host checkpoint commit and sustained failure qualification required |
| SC-5 | Bounded cell slots, connections, queues, frames, staging and deadlines; tenant turns; no uncertain auto-replay | Partial: ingress/handshake-rate controls, network isolation, adversarial load tests and capacity planning required |
| IR-4, SI-4 | Visible failed/cancelled/uncertain states, explicit repair events and backend quarantine after uncertain accelerator failure | Partial: incident response staffing, independent monitoring and operational drills required |

Concrete fixtures: `test_audit.py`, `test_scheduler.py`, `test_fleet_service.py`,
`test_connections.py`, `test_erasure.py`, `test_jax_llm.py`, native
`hyperl_erasure_contract`, and `qualify_llm.py`. The native source also runs through
ASan/UBSan where supported. See [validation](VALIDATION.md) for current passes,
skips, initial failures and actual hardware scope.

## Admission and stop behavior

A cell lease/epoch and exact profile are checked before execution and result
publication. Changed/revoked instances, expired leases, unavailable devices and
clock rollback reject. Missing TPU hardware cannot substitute a CPU. Native
model subprocesses and private API keys are owned by the session. Warm profile
reuse retains explicit context and reevaluates each request. Tensor/model kernels
have no tool-execution, shell, radio, actuator or privilege operation in their
request schema.

Cancellation requests and true termination are distinct. The native process can
be stopped locally; upstream RPC/device completion can remain uncertain. JAX
checks cancellation at host/token boundaries and cannot promise immediate device
interruption. Uncertain accelerator failures quarantine the agent. No automatic
replay or partially verified result is accepted. Independent stop, power-loss,
thermal, multi-host partition and operator-recovery tests are still deployment gates.

## Assessment boundary

This baseline does not assert a defence procurement standard, classified data
handling approval, FIPS/Common Criteria validation, air-gap accreditation or
production reliability. Establish the actual system boundary, threat model,
classification/data policy, approved cryptographic providers, supply-chain process,
platform security, physical controls and independent assessment before applying
those labels. HyperL can supply measured software evidence and explicit failure
contracts; deployment assurance belongs to the evaluated whole system.
