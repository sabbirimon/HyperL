// SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
// Copyright 2026 Sabbir Hassan Imon and respective contributors.
package org.hyperl.distributed

import org.hyperl.MemoryPlanner
import kotlinx.serialization.Serializable
import java.nio.file.*

@Serializable data class MemoryDomain(val id: String, val kind: String, val capacityBytes: Long? = null,
    val freeBytes: Long? = null, val admissionBytes: Long? = null, val sharedWith: String? = null,
    val technology: String? = null, val source: String, val observedAtEpochMs: Long = System.currentTimeMillis(),
    val physicalReservation: Boolean = false, val recommendedWorkingSetBytes: Long? = null,
    val maxAllocationBytes: Long? = null)

/** Observations are per domain. Filesystem capacity is never included in RAM or shard admission. */
object MemoryDomains {
    fun observe(budget: Long, scratch: Path? = null): List<MemoryDomain> {
        val snapshot = MemoryPlanner.observe()
        val result = mutableListOf(
            MemoryDomain("system", "SYSTEM_RAM", snapshot.environmentTotalBytes, snapshot.environmentFreeBytes, source = "JDK operating environment; may be container scoped"),
            MemoryDomain("heap", "JVM_HEAP", snapshot.heapLimitBytes, snapshot.heapLimitBytes - snapshot.heapUsedBytes,
                minOf(budget, snapshot.availableBytes()), sharedWith = "system", source = "JDK heap and configured application budget"))
        if (System.getProperty("os.name").startsWith("Linux")) {
            for (index in 0 until 256) {
                val path = Path.of("/sys/devices/system/node/node$index/meminfo")
                if (!Files.isRegularFile(path)) continue
                runCatching {
                    val text = Files.newInputStream(path).use { it.readNBytes(16385) }.also { require(it.size <= 16384) }.toString(Charsets.UTF_8)
                    fun field(name: String): Long? = Regex("Node $index $name:\\s+(\\d+) kB").find(text)?.groupValues?.get(1)?.toLongOrNull()?.takeIf { it <= Long.MAX_VALUE / 1024 }?.times(1024)
                    result += MemoryDomain("numa-$index", "NUMA_RAM", field("MemTotal"), field("MemFree"), sharedWith = "system", source = "Linux node$index/meminfo; subset of system RAM")
                }
            }
        }
        scratch?.let {
            require(it.isAbsolute && Files.isDirectory(it, LinkOption.NOFOLLOW_LINKS)) { "Owner-selected existing storage directory required" }
            val store = Files.getFileStore(it)
            result += MemoryDomain("storage", "FILESYSTEM_STORAGE", store.totalSpace, store.usableSpace,
                source = "FileStore at owner-selected directory; filesystem type ${store.type()}; disk technology unknown")
        }
        return result
    }
}
