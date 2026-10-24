package org.hyperl

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.Path
import java.nio.file.StandardOpenOption
import java.security.InvalidKeyException
import java.security.MessageDigest
import java.security.PrivateKey
import java.security.PublicKey
import java.security.Signature
import java.security.SignatureException

/** Authenticated metadata. Only [HyperLArtifactSignatures.verify] also checks artifact bytes. */
data class HyperLArtifactManifest(
    val artifactId: String,
    val context: String,
    val bytes: Long,
    val sha256: String,
)

/**
 * Bounded Java 17 JCA Ed25519 signatures over a canonical SHA-256 artifact manifest.
 * Keys and their trust decisions belong to the caller; no key material is encoded here.
 * Successful verification establishes integrity relative to that key, not freshness,
 * job authorization, confidentiality, code safety or subsequent immutability of a path.
 * The caller must retain/control the verified artifact before using it.
 *
 * Wire v1, big-endian: ASCII HLAS, version u8=1, signature algorithm u8=1 (Ed25519),
 * digest algorithm u8=1 (SHA-256), artifact ID length u16 + ASCII ID, context length
 * u16 + ASCII context, nonnegative size i64, raw digest[32], signature[64]. Every
 * byte before the signature is signed, including version/algorithm/domain/context.
 * IDs and contexts are nonempty, visible ASCII without whitespace; normalization,
 * extension fields and trailing bytes are not accepted. Other versions/algorithms
 * need an explicit future codec and never fall back to this implementation.
 */
object HyperLArtifactSignatures {
    const val MAX_ARTIFACT_BYTES = 4L * 1024 * 1024 * 1024 * 1024
    const val MAX_ENVELOPE_BYTES = 1024
    private const val BUFFER_BYTES = 64 * 1024
    private const val SIGNATURE_BYTES = 64
    private const val DIGEST_BYTES = 32
    private const val VERSION = 1
    private const val ED25519 = 1
    private const val SHA256 = 1
    private val magic = byteArrayOf(0x48, 0x4c, 0x41, 0x53)

    /** Reads an existing regular file in bounded chunks; never creates or changes keys/files. */
    suspend fun sign(
        source: Path,
        artifactId: String,
        context: String,
        key: PrivateKey,
        maxBytes: Long,
    ): ByteArray {
        label(artifactId, 128)
        label(context, 256)
        quota(maxBytes)
        // Reject unsupported keys before reading potentially large artifacts. JCA owns
        // curve/key validation; this also supports provider-owned opaque key handles.
        val signer = Signature.getInstance("Ed25519")
        try { signer.initSign(key) } catch (e: InvalidKeyException) {
            throw IllegalArgumentException("An Ed25519 signing key is required", e)
        }
        val observed = hash(source, maxBytes)
        currentCoroutineContext().ensureActive()
        val payload = encode(HyperLArtifactManifest(artifactId, context, observed.bytes, observed.digest.hex()))
        signer.update(payload)
        val signature = signer.sign()
        check(signature.size == SIGNATURE_BYTES) { "Unexpected Ed25519 signature size" }
        currentCoroutineContext().ensureActive()
        return payload + signature
    }

    /**
     * Verifies metadata against the caller's trusted key/context, then streams the artifact
     * and checks its exact length and digest. No output is published or code executed.
     * Cancellation is checked between bounded reads; blocking filesystem/JCA calls are not
     * promised to be interruptible. The source's final path component cannot be a symlink.
     */
    suspend fun verify(
        source: Path,
        envelope: ByteArray,
        trustedKey: PublicKey,
        expectedArtifactId: String,
        expectedContext: String,
        maxBytes: Long,
    ): HyperLArtifactManifest {
        currentCoroutineContext().ensureActive()
        val manifest = verifyManifest(envelope, trustedKey, expectedArtifactId, expectedContext, maxBytes)
        val observed = hash(source, manifest.bytes)
        if (observed.bytes != manifest.bytes || !MessageDigest.isEqual(observed.digest, unhex(manifest.sha256))) {
            throw SecurityException("Artifact size or SHA-256 does not match the signed manifest")
        }
        currentCoroutineContext().ensureActive()
        return manifest
    }

    /**
     * METADATA ONLY: validates the envelope signature, identity, context and declared size.
     * Does not read, hash or attest the actual artifact. Use [verify] before consuming bytes.
     * Invalid encoding/limits throw IllegalArgumentException; a trust/context/signature
     * mismatch throws SecurityException. No trust is derived from an embedded public key.
     */
    fun verifyManifest(
        envelope: ByteArray,
        trustedKey: PublicKey,
        expectedArtifactId: String,
        expectedContext: String,
        maxBytes: Long,
    ): HyperLArtifactManifest {
        label(expectedArtifactId, 128)
        label(expectedContext, 256)
        quota(maxBytes)
        require(envelope.size in 117..MAX_ENVELOPE_BYTES) { "Invalid artifact signature envelope size" }
        // Bound first, then take ownership so callers cannot race parsing/signature checks.
        val owned = envelope.copyOf()
        val input = ByteBuffer.wrap(owned).order(ByteOrder.BIG_ENDIAN)
        require(ByteArray(magic.size).also(input::get).contentEquals(magic)) { "Unknown artifact signature format" }
        require(input.get().toInt() == VERSION) { "Unsupported artifact signature version" }
        require(input.get().toInt() == ED25519) { "Unsupported artifact signature algorithm" }
        require(input.get().toInt() == SHA256) { "Unsupported artifact digest algorithm" }
        fun readLabel(max: Int): String {
            require(input.remaining() >= 2) { "Truncated artifact signature envelope" }
            val size = input.short.toInt() and 0xffff
            require(size in 1..max && input.remaining() >= size) { "Invalid artifact metadata length" }
            val bytes = ByteArray(size).also(input::get)
            require(bytes.all { (it.toInt() and 0xff) in 0x21..0x7e }) { "Artifact metadata must be visible ASCII" }
            return String(bytes, Charsets.US_ASCII)
        }
        val id = readLabel(128)
        val context = readLabel(256)
        require(input.remaining() == 8 + DIGEST_BYTES + SIGNATURE_BYTES) { "Truncated or trailing artifact signature data" }
        val size = input.long
        require(size in 0..maxBytes) { "Signed artifact exceeds the caller's byte limit" }
        val digest = ByteArray(DIGEST_BYTES).also(input::get)
        val signature = ByteArray(SIGNATURE_BYTES).also(input::get)
        val manifest = HyperLArtifactManifest(id, context, size, digest.hex())
        if (id != expectedArtifactId || context != expectedContext) {
            throw SecurityException("Signed artifact identity or context does not match")
        }
        val verifier = Signature.getInstance("Ed25519")
        try { verifier.initVerify(trustedKey) } catch (e: InvalidKeyException) {
            throw IllegalArgumentException("A trusted Ed25519 verification key is required", e)
        }
        verifier.update(owned, 0, owned.size - SIGNATURE_BYTES)
        val valid = try { verifier.verify(signature) } catch (_: SignatureException) { false }
        if (!valid) throw SecurityException("Artifact manifest signature verification failed")
        return manifest
    }

    private data class Observed(val bytes: Long, val digest: ByteArray)

    private suspend fun hash(source: Path, maxBytes: Long): Observed = withContext(Dispatchers.IO) {
        quota(maxBytes)
        currentCoroutineContext().ensureActive()
        require(Files.isRegularFile(source, LinkOption.NOFOLLOW_LINKS)) { "Artifact must be an existing regular file" }
        val digest = MessageDigest.getInstance("SHA-256")
        var count = 0L
        Files.newByteChannel(source, setOf(StandardOpenOption.READ, LinkOption.NOFOLLOW_LINKS)).use { channel ->
            require(channel.size() <= maxBytes) { "Artifact exceeds the caller's byte limit" }
            val buffer = ByteBuffer.allocate(BUFFER_BYTES)
            while (true) {
                currentCoroutineContext().ensureActive()
                buffer.clear()
                val read = channel.read(buffer)
                if (read < 0) break
                require(read.toLong() <= maxBytes - count) { "Artifact exceeds the caller's byte limit" }
                buffer.flip()
                digest.update(buffer)
                count += read
            }
        }
        currentCoroutineContext().ensureActive()
        Observed(count, digest.digest())
    }

    private fun encode(manifest: HyperLArtifactManifest): ByteArray {
        val bytes = ByteArrayOutputStream()
        DataOutputStream(bytes).use { output ->
            output.write(magic)
            output.writeByte(VERSION)
            output.writeByte(ED25519)
            output.writeByte(SHA256)
            for (value in listOf(manifest.artifactId, manifest.context)) {
                val text = value.toByteArray(Charsets.US_ASCII)
                output.writeShort(text.size)
                output.write(text)
            }
            output.writeLong(manifest.bytes)
            output.write(unhex(manifest.sha256))
        }
        return bytes.toByteArray()
    }

    private fun label(value: String, max: Int) {
        require(value.length in 1..max && value.all { it.code in 0x21..0x7e }) { "Artifact metadata must be bounded, visible ASCII" }
    }

    private fun quota(value: Long) { require(value in 0..MAX_ARTIFACT_BYTES) { "Invalid artifact byte limit" } }
    private fun ByteArray.hex(): String = joinToString("") { "%02x".format(it.toInt() and 0xff) }
    private fun unhex(value: String): ByteArray = ByteArray(DIGEST_BYTES) { value.substring(it * 2, it * 2 + 2).toInt(16).toByte() }
}
