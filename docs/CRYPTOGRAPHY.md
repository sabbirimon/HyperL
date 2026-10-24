# Artifact signatures and encrypted datasets

Updated 2026-10-24. This guide describes the implemented JVM API and local dataset
CLI. Cryptography is separate from HyperL's f32 program language and sharding
planner. No CUDA/tensor backend, distributed trust service, crypto accelerator or
formal certification is implied.

## Which operation to use

| Need | Implemented operation | What it establishes |
| --- | --- | --- |
| Detect a byte change | Streamed SHA-256 | A digest comparison; an untrusted digest alone does not establish origin |
| Authenticate artifact metadata and bytes | `HyperLArtifactSignatures.sign` / `verify`, Ed25519 plus SHA-256 | Integrity relative to an independently trusted public key and expected identity/context |
| Inspect signed metadata only | `verifyManifest` | Signature, expected identity/context and declared-size validation; artifact bytes are not read |
| Encrypt a local dataset | `keygen`, `data-import`, `data-export`, AES-256-GCM | Confidentiality and integrity of the stored chunks/manifest relative to the secret dataset key |

Signatures do not encrypt artifacts. AES-GCM dataset keys are not Ed25519 keys.
Neither operation starts a worker, installs a plugin or authorizes execution.

## Ed25519 artifact API

`org.hyperl.HyperLArtifactSignatures` uses Java 17's JCA `Ed25519` and `SHA-256`
implementations. The caller supplies key objects and retains responsibility for
storage, provenance, rotation, revocation and authorization. Keys are not encoded
in the signature envelope, language JSON or result manifest. There is currently
no signing-key generation/import CLI or automatic signing at installation.

The following Kotlin function illustrates a local sign/verify round trip. The
provided private key belongs to the publisher; the public key must come from the
consumer's independently approved trust configuration. Production publishing and
consumption are normally separate operations.

```kotlin
import java.nio.file.Path
import java.security.PrivateKey
import java.security.PublicKey
import org.hyperl.HyperLArtifactManifest
import org.hyperl.HyperLArtifactSignatures

suspend fun verifyOwnedArtifact(
    source: Path,
    publisherKey: PrivateKey,
    trustedPublisherKey: PublicKey
): HyperLArtifactManifest {
    val id = "adapter/cpu-v1"
    val context = "hyperl:adapter:linux-x86_64:release-1"
    val quota = 64L * 1024 * 1024 // Caller-selected 64 MiB limit.
    val envelope = HyperLArtifactSignatures.sign(
        source, id, context, publisherKey, quota
    )
    return HyperLArtifactSignatures.verify(
        source, envelope, trustedPublisherKey, id, context, quota
    )
}
```

For downloaded content, pass the received envelope to `verify` with your trusted
key, expected artifact ID/context and a local quota. Never trust a public key just
because it arrived beside the artifact. `verifyManifest` is useful for rejecting
bad metadata before transferring a large file, but cannot substitute for `verify`
before consuming the file. Success returns `artifactId`, `context`, `bytes` and
`sha256`; it neither publishes output nor runs code.

### Limits and failure behavior

- Caller quota: 0 through 4 TiB. Zero permits an empty artifact. The maximum is
  a format/API bound, not multi-terabyte throughput evidence.
- Files are read in 64 KiB chunks; hashing does not load the whole artifact into
  RAM. Existing regular files are required and the final path component cannot
  be a symbolic link. Trusted parent directories remain the caller's concern.
- Artifact IDs use 1–128 visible ASCII characters; contexts use 1–256. Whitespace,
  control characters and non-ASCII metadata are rejected. Labels contain no keys
  and should contain no sensitive personal information.
- Envelopes are capped at 1,024 bytes. Unsupported version/algorithm, malformed
  lengths, extra trailing bytes and quota violations fail closed.
- Ed25519 signatures are 64 bytes and SHA-256 digests are 32 bytes. Wrong keys,
  unexpected identity/context, modified signatures or file content are rejected.
- Cancellation is checked between bounded reads and before returning success.
  Blocking filesystem and JCA calls are not promised to stop immediately.

An invalid format/limit/key type generally throws `IllegalArgumentException`;
signature, trust/context or exact-byte mismatches throw `SecurityException`.
Filesystem and provider failures can propagate. Callers must retain failure and
cancellation states rather than interpreting missing output as success.

### Versioned envelope

Wire v1 is big-endian: `HLAS` magic; version byte `1`; signature algorithm byte
`1` (Ed25519); digest algorithm byte `1` (SHA-256); unsigned 16-bit length plus
ASCII artifact ID; unsigned 16-bit length plus ASCII context; nonnegative signed
64-bit artifact size; raw 32-byte digest; and the 64-byte signature. Every preceding
byte is signed. No alternate algorithm, normalization or unknown extension is
silently accepted. A future protocol needs a new explicit codec/compatibility gate.

### Trust, replay and file lifetime

The envelope contains no expiry, nonce, monotonic version or anti-replay ledger.
A valid old artifact can verify again. Callers must enforce approved releases,
expiry/revocation and replay rules independently; an expected release-specific
context helps only when that expectation comes from trusted policy. Verification
does not prove the publisher is currently authorized, the artifact is safe, or
the workload is permitted on a cluster.

The signed digest covers the bytes observed while reading. It does not lock the
path or prevent another process from replacing/changing that file later. Verify
inside caller-controlled storage and retain control of the exact artifact used;
do not verify an attacker-writable path and later reopen it for execution. No
snapshot/immutable handle, network transport, KMS/HSM integration or secure-memory
erasure is provided by this API.

## Existing AES-256-GCM dataset storage

Use distinct private dataset paths and a securely stored key:

```sh
hyperl keygen /private/keys/dataset.key
hyperl data-import /data/source.bin /data/dataset /private/keys/dataset.key 10737418240
hyperl data-export /data/dataset /data/restored.bin /private/keys/dataset.key 10737418240
```

`keygen` creates a random 32-byte AES key without overwriting. Import/export use
an explicit byte quota, up to 4 TiB. Payloads stream in 4 MiB AES-GCM chunks with
SHA-256 integrity checks; the `HLM1` manifest is also encrypted/authenticated in
bounded records of at most 512 plaintext bytes and an authenticated final footer.
An export publishes its new destination only after complete verification.
Existing destinations and incomplete/tampered outputs are rejected. Abrupt process
or OS failure may leave private staging that requires owner review.

Fresh random chunk nonces do not remove the need for key lifecycle management.
Use dataset-specific keys, back them up privately and plan rotation before very
large encryption volumes. Key rotation, KMS, secure erasure and distributed key
exchange are not implemented. POSIX-created key files/directories are private;
review permissions/ACLs on the actual host. Never put secret keys in prompts,
program JSON, source, CLI literal arguments or logs. Key-file paths can appear in
local process history and should themselves avoid sensitive identifiers.

## Evidence boundaries

Tests cover real JCA signatures, bounded multi-chunk hashing, independently checked
signature framing, wrong keys/contexts, modified files, malformed envelopes,
quotas and cancellation. Dataset tests cover AES-GCM round trips and corruption
rejection. Consult [validation](VALIDATION.md) for runs actually completed and
their exact source scope. Test coverage is not a security audit, hardware-provider
qualification, side-channel evaluation or certification. General crypto operators,
complex-number math, full model sharding and secure cluster transport remain
separate implementation milestones.
