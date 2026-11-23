package org.hyperl

import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.Assert.*
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.file.Files
import java.nio.file.Path
import java.security.KeyPair
import java.security.KeyPairGenerator
import java.security.Signature
import java.util.concurrent.CancellationException

class HyperLArtifactSignaturesTest {
    private val id = "adapter/cpu-v1"
    private val context = "hyperl:adapter:linux-x86_64:release-1"
    private fun keys(): KeyPair = KeyPairGenerator.getInstance("Ed25519").generateKeyPair()
    private fun <T> artifact(bytes: ByteArray, work: (Path) -> T): T {
        val file = Files.createTempFile("hyperl-signature-", ".bin")
        return try { Files.write(file, bytes); work(file) } finally { Files.deleteIfExists(file) }
    }

    @Test fun realEd25519SignatureBindsMultiChunkArtifactAndCanonicalManifest() = runBlocking {
        val key = keys()
        val data = ByteArray(160_000) { (it % 251).toByte() }
        artifact(data) { file ->
            val first = runBlocking { HyperLArtifactSignatures.sign(file, id, context, key.private, data.size.toLong()) }
            val second = runBlocking { HyperLArtifactSignatures.sign(file, id, context, key.private, data.size.toLong()) }
            assertArrayEquals("Canonical metadata and Ed25519 produce stable signatures", first, second)
            assertTrue(first.size <= HyperLArtifactSignatures.MAX_ENVELOPE_BYTES)
            val result = runBlocking { HyperLArtifactSignatures.verify(file, first, key.public, id, context, data.size.toLong()) }
            assertEquals(data.size.toLong(), result.bytes)
            assertEquals(id, result.artifactId)
            assertEquals(context, result.context)
            assertEquals(java.security.MessageDigest.getInstance("SHA-256").digest(data).joinToString("") { "%02x".format(it.toInt() and 255) }, result.sha256)
            // Independent JCA verification demonstrates standard Ed25519, not a custom primitive.
            val jca = Signature.getInstance("Ed25519")
            jca.initVerify(key.public)
            jca.update(first, 0, first.size - 64)
            assertTrue(jca.verify(first.copyOfRange(first.size - 64, first.size)))
        }
    }

    @Test fun mutationWrongTrustedKeyAndWrongIdentityOrContextAreRejected() {
        val key = keys()
        artifact(byteArrayOf(1, 2, 3)) { file ->
            val signed = runBlocking { HyperLArtifactSignatures.sign(file, id, context, key.private, 3) }
            assertThrows(SecurityException::class.java) { HyperLArtifactSignatures.verifyManifest(signed, keys().public, id, context, 3) }
            assertThrows(SecurityException::class.java) { HyperLArtifactSignatures.verifyManifest(signed, key.public, "another-artifact", context, 3) }
            assertThrows(SecurityException::class.java) { HyperLArtifactSignatures.verifyManifest(signed, key.public, id, "$context-other", 3) }
            // Mutate digest, size, signature, and metadata without changing the overall framing.
            for (offset in listOf(signed.size - 65, signed.size - 64 - 32 - 1, signed.lastIndex, 9)) {
                val changed = signed.copyOf().also { it[offset] = (it[offset].toInt() xor 1).toByte() }
                assertThrows(SecurityException::class.java) { HyperLArtifactSignatures.verifyManifest(changed, key.public, id, context, 10) }
            }
            Files.write(file, byteArrayOf(1, 2, 4))
            assertThrows(SecurityException::class.java) { runBlocking { HyperLArtifactSignatures.verify(file, signed, key.public, id, context, 3) } }
            // A correctly signed manifest is metadata only; it does not attest the changed file.
            assertEquals(3L, HyperLArtifactSignatures.verifyManifest(signed, key.public, id, context, 3).bytes)
            Files.write(file, byteArrayOf(1, 2))
            assertThrows(SecurityException::class.java) { runBlocking { HyperLArtifactSignatures.verify(file, signed, key.public, id, context, 3) } }
            Files.write(file, byteArrayOf(1, 2, 3, 4))
            assertThrows(IllegalArgumentException::class.java) { runBlocking { HyperLArtifactSignatures.verify(file, signed, key.public, id, context, 4) } }
        }
    }

    @Test fun malformedTruncatedTrailingUnsupportedAndOversizedEnvelopesFailClosed() {
        val key = keys()
        artifact(byteArrayOf(1)) { file ->
            val signed = runBlocking { HyperLArtifactSignatures.sign(file, id, context, key.private, 1) }
            val invalid = listOf(
                byteArrayOf(), signed.copyOf(signed.size - 1), signed + byteArrayOf(0),
                ByteArray(HyperLArtifactSignatures.MAX_ENVELOPE_BYTES + 1),
                signed.copyOf().also { it[0] = 0 },
                signed.copyOf().also { it[4] = 2 }, // unknown version
                signed.copyOf().also { it[5] = 2 }, // unknown signature algorithm
                signed.copyOf().also { it[6] = 2 }, // unknown digest algorithm
                signed.copyOf().also { it[7] = 0x7f; it[8] = 0x7f }, // oversized label
                signed.copyOf().also { it[9] = 0 }, // noncanonical control byte
                signed.copyOf().also { it[9] = 0xff.toByte() }, // not ASCII
                signed.copyOf().also { ByteBuffer.wrap(it).putLong(it.size - 64 - 32 - 8, -1L) },
            )
            invalid.forEach { envelope ->
                assertThrows(IllegalArgumentException::class.java) { HyperLArtifactSignatures.verifyManifest(envelope, key.public, id, context, 1) }
            }
            for (length in signed.indices) {
                assertThrows(IllegalArgumentException::class.java) { HyperLArtifactSignatures.verifyManifest(signed.copyOf(length), key.public, id, context, 1) }
            }
        }
    }

    @Test fun quotasMetadataBoundsAndUnsupportedKeysAreRejected() {
        val key = keys()
        artifact(byteArrayOf(1, 2)) { file ->
            for (quota in listOf(-1L, HyperLArtifactSignatures.MAX_ARTIFACT_BYTES + 1, 1L)) {
                assertThrows(IllegalArgumentException::class.java) { runBlocking { HyperLArtifactSignatures.sign(file, id, context, key.private, quota) } }
            }
            for (bad in listOf("", "a".repeat(129), "has space", "line\nfeed", "日本語")) {
                assertThrows(IllegalArgumentException::class.java) { runBlocking { HyperLArtifactSignatures.sign(file, bad, context, key.private, 2) } }
            }
            assertThrows(IllegalArgumentException::class.java) { runBlocking { HyperLArtifactSignatures.sign(file, id, "x".repeat(257), key.private, 2) } }
            val wrongCurve = KeyPairGenerator.getInstance("Ed448").generateKeyPair()
            assertThrows(IllegalArgumentException::class.java) { runBlocking { HyperLArtifactSignatures.sign(file, id, context, wrongCurve.private, 2) } }
            val signed = runBlocking { HyperLArtifactSignatures.sign(file, id, context, key.private, 2) }
            assertThrows(IllegalArgumentException::class.java) { HyperLArtifactSignatures.verifyManifest(signed, wrongCurve.public, id, context, 2) }
            assertThrows(IllegalArgumentException::class.java) { HyperLArtifactSignatures.verifyManifest(signed, key.public, id, context, 1) }
        }
    }

    @Test fun emptyArtifactAndMaximumLabelsWorkWithoutRequiringPositiveFileSize() {
        val key = keys()
        artifact(byteArrayOf()) { file ->
            val longId = "a".repeat(128)
            val longContext = "c".repeat(256)
            val signed = runBlocking { HyperLArtifactSignatures.sign(file, longId, longContext, key.private, 0) }
            val result = runBlocking { HyperLArtifactSignatures.verify(file, signed, key.public, longId, longContext, 0) }
            assertEquals(0L, result.bytes)
            assertEquals("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", result.sha256)
        }
    }

    @Test fun cancellationDoesNotReturnSuccessfulSignatureAndDirectoriesAreRejected() = runBlocking {
        val key = keys()
        artifact(byteArrayOf(1)) { file ->
            assertThrows(CancellationException::class.java) {
                runBlocking { withContext(Job().also { it.cancel() }) { HyperLArtifactSignatures.sign(file, id, context, key.private, 1) } }
            }
        }
        val directory = Files.createTempDirectory("hyperl-signature-directory-")
        try {
            assertThrows(IllegalArgumentException::class.java) { runBlocking { HyperLArtifactSignatures.sign(directory, id, context, key.private, 0) } }
        } finally { Files.deleteIfExists(directory) }
        Unit
    }
}
