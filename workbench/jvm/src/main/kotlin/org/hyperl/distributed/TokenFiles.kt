// SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
// Copyright 2026 Sabbir Hassan Imon and respective contributors.
package org.hyperl.distributed
import java.nio.ByteBuffer
import java.nio.file.*
import java.nio.file.attribute.PosixFilePermissions
import java.security.SecureRandom
import java.util.Base64
object TokenFiles {
    fun create(path: Path) {
        require(path.isAbsolute && Files.isDirectory(path.parent)) { "Existing private parent and absolute destination required" }
        val secret = ByteArray(32).also { SecureRandom().nextBytes(it) }
        val bytes = (Base64.getUrlEncoder().withoutPadding().encodeToString(secret) + "\n").toByteArray(Charsets.US_ASCII)
        val attributes = if (Files.getFileStore(path.parent).supportsFileAttributeView("posix"))
            arrayOf(PosixFilePermissions.asFileAttribute(PosixFilePermissions.fromString("rw-------"))) else emptyArray()
        try { Files.newByteChannel(path, setOf(StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE), *attributes).use { channel ->
            val buffer = ByteBuffer.wrap(bytes); while (buffer.hasRemaining()) channel.write(buffer)
        } } finally { secret.fill(0); bytes.fill(0) }
    }
}
