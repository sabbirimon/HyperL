// SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
// Copyright 2026 Sabbir Hassan Imon and respective contributors.
package org.hyperl

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.Path
import java.nio.file.StandardOpenOption
import java.security.MessageDigest

/** Explicit JCA algorithms; no aliases, custom primitives or fallback algorithm. */
enum class HyperLHashAlgorithm(val identifier: String, val digestBytes: Int) {
    SHA256("SHA-256", 32), SHA384("SHA-384", 48), SHA512("SHA-512", 64);

    companion object {
        fun fromIdentifier(value: String): HyperLHashAlgorithm =
            entries.firstOrNull { it.identifier == value }
                ?: throw IllegalArgumentException("Unsupported artifact hash algorithm")
    }
}

data class HyperLArtifactHash(val algorithm: HyperLHashAlgorithm, val bytes: Long, val hexDigest: String)

/** Separate file-integrity API. HLAS v1 signatures and HLM2 datasets remain unchanged.
 * A hash is not a signature, encryption, authentication, authorization or proof of
 * freshness. Caller-controlled storage is required: this does not lock the path or
 * make the observed file immutable after reading. Parent paths must be trusted.
 */
object HyperLHashes {
    const val MAX_ARTIFACT_BYTES = 4L * 1024 * 1024 * 1024 * 1024
    private const val BUFFER_BYTES = 64 * 1024

    /** Streams one regular file with an explicit quota (zero allows an empty file).
     * Cancellation is checked between reads and before returning; blocking filesystem
     * and JCA calls are not promised to be interruptible. The final path component
     * cannot be a symlink. Missing JCA algorithms fail without substituting another.
     */
    suspend fun hash(source: Path, maxBytes: Long,
        algorithm: HyperLHashAlgorithm = HyperLHashAlgorithm.SHA256): HyperLArtifactHash = withContext(Dispatchers.IO) {
        require(maxBytes in 0..MAX_ARTIFACT_BYTES) { "Invalid artifact byte limit" }
        currentCoroutineContext().ensureActive()
        require(Files.isRegularFile(source, LinkOption.NOFOLLOW_LINKS)) { "Artifact must be an existing regular file" }
        val digest = MessageDigest.getInstance(algorithm.identifier)
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
        val raw = digest.digest()
        check(raw.size == algorithm.digestBytes) { "Unexpected JCA digest length" }
        val hex = raw.joinToString("") { "%02x".format(it.toInt() and 255) }
        currentCoroutineContext().ensureActive()
        HyperLArtifactHash(algorithm, count, hex)
    }
}
