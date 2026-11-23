// SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
// Copyright 2026 Sabbir Hassan Imon and respective contributors.
package org.hyperl

import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.Assert.*
import org.junit.Assume.assumeNoException
import org.junit.Test
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.CancellationException

class HyperLHashesTest {
    private fun <T> file(bytes: ByteArray, action: (Path) -> T): T {
        val path = Files.createTempFile("hyperl-hash-", ".bin")
        return try { Files.write(path, bytes); action(path) } finally { Files.deleteIfExists(path) }
    }

    @Test fun independentStandardAbcVectorsAndDefaultSha256() {
        // Published SHA-2 known answers, not generated through the implementation under test.
        val expected = mapOf(
            HyperLHashAlgorithm.SHA256 to "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            HyperLHashAlgorithm.SHA384 to "cb00753f45a35e8bb5a03d699ac65007272c32ab0eded1631a8b605a43ff5bed8086072ba1e7cc2358baeca134c825a7",
            HyperLHashAlgorithm.SHA512 to "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f")
        file("abc".toByteArray(Charsets.US_ASCII)) { path ->
            for ((algorithm, answer) in expected) {
                val result = runBlocking { HyperLHashes.hash(path, 3, algorithm) }
                assertEquals(algorithm, result.algorithm)
                assertEquals(3L, result.bytes)
                assertEquals(answer, result.hexDigest)
            }
            assertEquals(expected.getValue(HyperLHashAlgorithm.SHA256), runBlocking { HyperLHashes.hash(path, 3) }.hexDigest)
        }
    }

    @Test fun millionAIndependentVectorsCrossManyReadBuffers() {
        val expected = mapOf(
            HyperLHashAlgorithm.SHA384 to "9d0e1809716474cb086e834e310a4a1ced149e9c00f248527972cec5704c2a5b07b8b3dc38ecc4ebae97ddd87f3d8985",
            HyperLHashAlgorithm.SHA512 to "e718483d0ce769644e2e42c7bc15b4638e1f98b13b2044285632a803afa973ebde0ff244877ea60a4cb0432ce577c31beb009c5c2c49aa2e4eadb217ad8cc09b")
        file(ByteArray(1_000_000) { 'a'.code.toByte() }) { path ->
            for ((algorithm, answer) in expected) {
                val result = runBlocking { HyperLHashes.hash(path, 1_000_000, algorithm) }
                assertEquals(1_000_000L, result.bytes)
                assertEquals(answer, result.hexDigest)
            }
        }
    }

    @Test fun emptyQuotaAndMutationAreObservedWithoutAlgorithmFallback() {
        file(byteArrayOf()) { path ->
            val result = runBlocking { HyperLHashes.hash(path, 0, HyperLHashAlgorithm.SHA512) }
            assertEquals(0L, result.bytes)
            assertEquals("cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e", result.hexDigest)
            Files.write(path, byteArrayOf(1))
            val changed = runBlocking { HyperLHashes.hash(path, 1, HyperLHashAlgorithm.SHA512) }
            assertNotEquals(result.hexDigest, changed.hexDigest)
            assertEquals(HyperLHashAlgorithm.SHA512, changed.algorithm)
        }
        for (name in listOf("SHA-1", "MD5", "sha-512", "SHA512", "", "SHA-384\n")) {
            assertThrows(IllegalArgumentException::class.java) { HyperLHashAlgorithm.fromIdentifier(name) }
        }
        for (algorithm in HyperLHashAlgorithm.entries) assertEquals(algorithm, HyperLHashAlgorithm.fromIdentifier(algorithm.identifier))
    }

    @Test fun invalidQuotasFilesAndCancelledWorkDoNotReturnHashes() {
        file(byteArrayOf(1, 2)) { path ->
            for (limit in listOf(-1L, 0L, 1L, HyperLHashes.MAX_ARTIFACT_BYTES + 1)) {
                assertThrows(IllegalArgumentException::class.java) { runBlocking { HyperLHashes.hash(path, limit, HyperLHashAlgorithm.SHA384) } }
            }
            assertThrows(CancellationException::class.java) {
                runBlocking { withContext(Job().also { it.cancel() }) { HyperLHashes.hash(path, 2, HyperLHashAlgorithm.SHA512) } }
            }
        }
        val directory = Files.createTempDirectory("hyperl-hash-invalid-")
        try {
            for (path in listOf(directory, directory.resolve("missing"))) {
                assertThrows(IllegalArgumentException::class.java) { runBlocking { HyperLHashes.hash(path, 0) } }
            }
        } finally { Files.deleteIfExists(directory) }
    }

    @Test fun finalComponentSymbolicLinkIsRejectedWhereHostPermitsLinks() {
        val directory = Files.createTempDirectory("hyperl-hash-link-")
        val target = directory.resolve("target")
        val link = directory.resolve("link")
        try {
            Files.write(target, byteArrayOf(1))
            try { Files.createSymbolicLink(link, target) } catch (error: Exception) { assumeNoException("Host does not permit symbolic links", error) }
            assertThrows(IllegalArgumentException::class.java) { runBlocking { HyperLHashes.hash(link, 1, HyperLHashAlgorithm.SHA384) } }
        } finally { Files.deleteIfExists(link); Files.deleteIfExists(target); Files.deleteIfExists(directory) }
    }
}
