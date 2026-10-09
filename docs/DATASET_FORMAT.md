# Local encrypted dataset formats

Alpha.6 writes **HLM2** and reads authenticated **HLM1**. Both use AES-256-GCM
with 12-byte random nonces and 128-bit tags, plus SHA-256 integrity checks.
Keys are 32 bytes from `SecureRandom`, not passwords. A password KDF is therefore
not part of this interface; no passphrase mode or KMS integration is claimed.
Keep keys in private files outside installations, source, programs and logs.

## Records and identity

The dataset has a canonical UUID in `id`, a `manifest.aesgcm`, and encrypted chunks.
HLM2 also has `key-id`: 64 lowercase hexadecimal SHA-256 characters derived from
the random AES key. This is a key selector/integrity identity, not a password hash
or key derivation function. Key ID, dataset UUID, format, chunk order and plaintext
length are authenticated with each chunk. Manifest record index is authenticated
separately. Altering an identity, nonce, ciphertext or authenticated field fails.

The manifest starts with ASCII `HLM1` or `HLM2`. Each subsequent frame consists of
a two-byte big-endian frame length, 12-byte nonce and GCM ciphertext/tag. Plaintext
JSON per frame is bounded to 512 bytes. HLM2 requires exactly:

```json
{"type":"chunk","index":0,"bytes":3,"sha256":"64 lowercase hexadecimal characters"}
{"type":"footer","totalBytes":3,"chunks":1,"sha256":"64 lowercase hexadecimal characters"}
```

The hash strings above describe the required shape, not valid fixture digests.
Numbers must be JSON integers, not numeric strings. Unknown fields/types, missing
fields, duplicate/out-of-order chunk indices, invalid hashes and trailing records
are rejected. HLM1 has the same chunk/footer fields without `type`; readers
recognize the exact field sets, never substrings in JSON text.

HLM1 chunks are `chunk_INDEX.aesgcm` in the dataset root. HLM2 uses
`chunks/GROUP/chunk_INDEX.aesgcm`, where GROUP is four hexadecimal digits for
`INDEX / 1024`. At most 1,024 chunks belong to a group. Each chunk is at most
4 MiB plaintext, and at most 1,048,576 chunks/4 TiB fit the format. These are
admission ceilings, not measured filesystem or multi-terabyte performance.

## Verification, publication and rotation

Export decrypts into private staging and checks per-chunk authentication/hash,
order, quota and the final authenticated totals/whole-file hash before publishing.
Final plaintext publication uses same-filesystem hard-link creation: destination
creation is atomic and fails if any destination exists. A filesystem without
hard links fails explicitly; there is no overwrite fallback. Cancellation before
publication and malformed/truncated input leave no accepted output.

Import/rekey reserve a new destination directory without replacing an existing
one. The authenticated manifest moves into it **last**, as the commit marker.
A reader cannot accept an incomplete reserved directory. This is not a claim of
one atomic whole-directory rename. Ordinary failures clean owned staging/reserved
directories; process/OS crashes can leave private incomplete directories for
owner inspection. No fsync durability, snapshot guarantee or secure disk erasure
is promised. Review private-parent Windows ACLs; POSIX-created keys/datasets have
private permissions.

```sh
hyperl keygen /private/keys/new.key
hyperl data-rekey /data/old /data/new /private/keys/old.key /private/keys/new.key 10737418240
hyperl data-export /data/new /data/restored.bin /private/keys/new.key 10737418240
```

Rotation requires a different key, verifies the full source and writes fresh UUID,
nonces, key identity and HLM2 ciphertext into a new dataset. Decrypted chunk buffers
are cleared after use; no intermediate plaintext file is exported during rotation.
The old dataset and key remain unchanged, including when rotating an HLM1 source.
Only the new key decrypts the new dataset. Java/provider copies may retain memory;
buffer clearing is not a whole-process secure-erasure guarantee. Back up the new
key and verify new output before deciding what to do with the old copies.

Independent legacy, multi-chunk rotation, authenticated malformed-record, wrong-key,
tamper, quota and no-overwrite tests exercise these boundaries. An independent
cryptographic/security review remains a separate deployment gate.
