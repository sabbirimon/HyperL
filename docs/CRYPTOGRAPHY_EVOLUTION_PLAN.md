<!-- SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0 -->
<!-- Copyright 2026 Sabbir Hassan Imon and respective contributors. -->
# Cryptography evolution and post-quantum migration

Reviewed 2026-10-24. Stronger options require an explicit threat model, protocol
and qualified implementation. No algorithm can promise permanent security.

## Implemented versus planned

| Purpose | Current HyperL implementation | Next qualified option |
| --- | --- | --- |
| Artifact digest | Bounded JCA SHA-256 default; explicit SHA-384 and SHA-512 through `HyperLHashes` | Policy-controlled digest selection in a new signed-envelope version |
| Artifact signature | HLAS v1: JCA Ed25519 over canonical metadata and SHA-256 | ML-DSA and a separately specified classical/PQ migration policy |
| Local dataset encryption | HLM2 AES-256-GCM with authenticated records and SHA-256 checks | Provider-backed key lifecycle; separately versioned key wrapping if needed |
| Network security | Existing configured JDK HTTPS and worker authentication | Qualified TLS provider and interoperable hybrid key establishment |
| Quantum computing | Separate circuit simulation/provider bridges | Does not implement post-quantum encryption, signatures or key establishment |

SHA-384/SHA-512 produce longer digests than SHA-256; they do not authenticate an
untrusted checksum or upgrade an Ed25519 signature into a post-quantum signature.
Choose a complete suite for its security target, interoperability, storage and
verification cost. [NIST Secure Hash Standard](https://csrc.nist.gov/pubs/fips/180-4/upd1/final).

## Standards to qualify

- **ML-KEM, FIPS 203:** establishes a shared secret; it is not a file-encryption
  or signature API. Evaluate an explicit parameter set and a reviewed protocol
  binding using its output with a KDF and authenticated encryption. See
  [FIPS 203](https://csrc.nist.gov/pubs/fips/203/final) and
  [NIST KEM guidance](https://csrc.nist.gov/pubs/sp/800/227/final).
- **ML-DSA, FIPS 204:** candidate for artifact/signing-key migration. Bind the
  exact parameter set, context, key encoding and signing mode; do not silently
  interchange pre-standard Dilithium formats or prehash modes.
  [FIPS 204](https://csrc.nist.gov/pubs/fips/204/final).
- **SLH-DSA, FIPS 205:** a distinct stateless hash-based signature option for
  cryptographic diversity. Qualify its exact parameter set and signature-size/
  latency costs; it is not a generic checksum and not a drop-in HLAS v1 signature.
  [FIPS 205](https://csrc.nist.gov/pubs/fips/205/final).

Track published errata and provider fixes. A standardized algorithm does not make
HyperL or an arbitrary installed library a validated cryptographic module.

## Compatibility and downgrade rules

1. Preserve HLAS v1 decoding and semantics: Ed25519, SHA-256 and its existing
   1,024-byte envelope ceiling. Never reinterpret its algorithm bytes or widen its
   parser opportunistically for a larger PQ signature.
2. Specify a new envelope version with canonical suite/parameter IDs, domain and
   context, exact digest/key/signature lengths and a separately bounded total size.
   All security-relevant fields must be authenticated; reject unknown/duplicate fields.
3. Select accepted suites from trusted local policy. Peer advertisements cannot
   lower the minimum suite, waive a required signature or extend a revoked key.
4. Missing provider, invalid key, unsupported mode or failed verification is an
   error. Never retry automatically with SHA-256, Ed25519 or a classical-only mode
   when the requested policy requires a stronger/different suite.
5. Keep read-old/write-new migration explicit and time-bounded by owner policy.
   Persist artifact provenance, key rotation/revocation and release freshness;
   signatures alone do not prevent replay or authorize execution.

For hybrid transport, use a reviewed provider's specified protocol construction,
not a home-made combination of secrets. OpenSSL 3.5 documents ML-KEM/ML-DSA/SLH-DSA
and hybrid TLS groups, but HyperL's JDK transport does not gain those capabilities
merely because OpenSSL is installed. [Provider algorithms](https://docs.openssl.org/3.5/man7/OSSL_PROVIDER-default/),
[TLS groups](https://docs.openssl.org/3.5/man3/SSL_CONF_cmd/).
Any dual-signature design must state which signatures are mandatory and bind both
to the same canonical artifact/context; accepting either signature is not the same
policy as requiring both. Interoperability and protocol review precede deployment.

## Delivery and acceptance gates

- Inventory signing, transport, storage, trust roots and data-retention lifetimes;
  prioritize migrations using those requirements. [NIST crypto-agility guidance](https://csrc.nist.gov/pubs/cswp/39/upd1/considerations-for-achieving-crypto-agility/final).
- Pin a maintained provider, version, license, platform/ABI and algorithm modes;
  keep external key handles and audited entropy paths, with no automatic downloads.
- Test independent standard vectors and cross-provider verification, malformed/
  oversized keys and envelopes, context mutation, downgrade attempts and replay policy.
- Measure signature/key/ciphertext bytes, peak memory, streaming digest cost and
  full verification/handshake time on actual supported devices. Bound allocations,
  concurrent operations, timeouts and cancellation without claiming interruptible JCA calls.
- Exercise staged key rotation, interrupted migrations and old/new reader policy;
  obtain independent protocol/security review before a production or validation claim.

Only the separate SHA-384/SHA-512 digest API is added in this increment. ML-KEM,
ML-DSA, SLH-DSA, hybrid TLS and new signature envelopes remain unimplemented.
See [current APIs and limits](CRYPTOGRAPHY.md) and [programming examples](PROGRAMMING_GUIDE.md).
