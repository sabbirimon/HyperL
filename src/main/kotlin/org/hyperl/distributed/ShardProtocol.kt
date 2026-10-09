// SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
// Copyright 2026 Sabbir Hassan Imon and respective contributors.
package org.hyperl.distributed

import org.hyperl.*
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.*
import java.net.URI
import java.nio.file.*
import java.security.MessageDigest
import java.util.UUID

/** Versioned, bounded control metadata and network-order IEEE f32 buffers. No executable input. */
object ShardProtocol {
    const val FORMAT = "hyperl-shard/1"
    const val CONTENT_TYPE = "application/vnd.hyperl.shard-v1"
    const val MAX_FRAME = 8 * 1024 * 1024
    const val MAX_METADATA = 96 * 1024
    const val MAX_WORKERS = 16
    const val MAX_SHARDS = 256
    val json = Json { encodeDefaults = true; ignoreUnknownKeys = false }
    val operations = setOf("add", "multiply", "relu")
    fun digest(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
    fun uuid(value: String) { require(value.length == 36 && UUID.fromString(value).toString() == value) { "Invalid job identity" } }
    fun origin(value: String): URI {
        require(value.length in 1..2048)
        val uri = URI(value)
        require(uri.host != null && uri.rawUserInfo == null && uri.rawQuery == null && uri.rawFragment == null && uri.path in listOf("", "/"))
        val host = uri.host.removePrefix("[").removeSuffix("]")
        require('%' !in host && (uri.port == -1 || uri.port in 1..65535))
        require(uri.scheme == "https" || uri.scheme == "http" && host in setOf("127.0.0.1", "::1")) { "Remote workers require trusted HTTPS; plaintext is numeric loopback only" }
        return uri
    }
    fun token(path: Path): String {
        require(path.isAbsolute && Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS) && Files.size(path) in 32..1024) { "Private absolute token file required" }
        val value = Files.readString(path).trim()
        require(value.length in 32..512 && value.all { it.code in 33..126 }) { "Invalid bearer token" }
        return value
    }
    fun shardBytes(program: HyperLProgram, elements: Int): Long {
        require(elements in 1..HyperLContract.MAX_VECTOR_ELEMENTS)
        // Request/response buffers, decoded arrays, CPU copies/intermediates, GPU staging and verification.
        return 512L * 1024 + 4L * elements * (5 * program.inputs.size + 3 * program.instructions.size + 8)
    }
    fun maxElements(program: HyperLProgram, credit: Long): Int {
        val perElement = 4L * (5 * program.inputs.size + 3 * program.instructions.size + 8)
        val retainedPerElement = program.inputs.size + program.instructions.size
        return minOf(((credit - 512L * 1024).coerceAtLeast(0) / perElement).coerceAtMost(Int.MAX_VALUE.toLong()).toInt(),
            HyperLContract.MAX_RETAINED_ELEMENTS / retainedPerElement, HyperLContract.MAX_VECTOR_ELEMENTS)
    }
    fun encodeRequest(meta: ShardRequest, inputs: Map<String, FloatArray>): ByteArray {
        validate(meta)
        require(inputs.keys == meta.program.inputs && inputs.values.all { it.size == meta.end - meta.start })
        return frame(json.encodeToString(meta)) { out -> meta.program.inputs.sorted().forEach { name -> inputs.getValue(name).forEach { value -> require(value.isFinite()); out.writeFloat(value) } } }
    }
    fun decodeRequest(bytes: ByteArray): Pair<ShardRequest, Map<String, FloatArray>> = unframe(bytes) { text, input ->
        val meta = json.decodeFromString<ShardRequest>(text); validate(meta)
        val count = meta.end - meta.start
        require(input.available().toLong() == 4L * count * meta.program.inputs.size) { "Shard input length mismatch" }
        meta to meta.program.inputs.sorted().associateWith { FloatArray(count) { input.readFloat().also { require(it.isFinite()) } } }
    }
    fun encodeResponse(meta: ShardResponse, output: FloatArray): ByteArray {
        require(output.size == meta.end - meta.start && output.all(Float::isFinite))
        return frame(json.encodeToString(meta)) { out -> output.forEach(out::writeFloat) }
    }
    fun decodeResponse(bytes: ByteArray): Pair<ShardResponse, FloatArray> = unframe(bytes) { text, input ->
        val meta = json.decodeFromString<ShardResponse>(text)
        require(meta.format == FORMAT && meta.end > meta.start && meta.start >= 0 && meta.end <= HyperLContract.MAX_VECTOR_ELEMENTS)
        require(input.available() == 4 * (meta.end - meta.start))
        meta to FloatArray(meta.end - meta.start) { input.readFloat().also { require(it.isFinite()) } }
    }
    fun checkJson(text: String, bound: Int = MAX_METADATA) {
        require(text.length <= bound)
        var depth = 0; var string = false; var escaped = false
        text.forEach { char -> if (string) { if (escaped) escaped = false else if (char == '\\') escaped = true else if (char == '"') string = false }
            else when (char) { '"' -> string = true; '{', '[' -> { depth++; require(depth <= 16) }; '}', ']' -> { depth--; require(depth >= 0) } } }
        require(depth == 0 && !string)
    }
    private fun validate(meta: ShardRequest) {
        require(meta.format == FORMAT); uuid(meta.jobId)
        require(meta.start >= 0 && meta.end > meta.start && meta.end <= HyperLContract.MAX_VECTOR_ELEMENTS)
        require(meta.timeoutMs in 100..30000)
        uuid(meta.workerInstanceId)
        meta.program.validate()
        require(meta.program.instructions.all { it.operation in operations }) { "Only elementwise worker operators are supported" }
        require((meta.program.inputs.size + meta.program.instructions.size).toLong() * (meta.end - meta.start) <= HyperLContract.MAX_RETAINED_ELEMENTS)
    }
    private fun frame(text: String, body: (DataOutputStream) -> Unit): ByteArray {
        val metadata = text.toByteArray(Charsets.UTF_8); require(metadata.size <= MAX_METADATA)
        val buffer = ByteArrayOutputStream()
        DataOutputStream(buffer).use { it.writeInt(0x484c5331); it.writeInt(metadata.size); it.write(metadata); body(it) }
        return buffer.toByteArray().also { require(it.size <= MAX_FRAME) }
    }
    private fun <T> unframe(bytes: ByteArray, body: (String, DataInputStream) -> T): T {
        require(bytes.size in 8..MAX_FRAME)
        return DataInputStream(ByteArrayInputStream(bytes)).use { input ->
            require(input.readInt() == 0x484c5331); val count = input.readInt()
            require(count in 1..MAX_METADATA && count <= input.available())
            val text = input.readNBytes(count).toString(Charsets.UTF_8)
            checkJson(text)
            body(text, input).also { require(input.available() == 0) { "Trailing shard data" } }
        }
    }
}

@Serializable data class ShardRequest(val format: String = ShardProtocol.FORMAT, val jobId: String, val start: Int, val end: Int,
    val program: HyperLProgram, val workerInstanceId: String, val timeoutMs: Long = 10000)
@Serializable data class ShardResponse(val format: String = ShardProtocol.FORMAT, val jobId: String, val start: Int, val end: Int,
    val requestSha256: String, val backend: HyperLTarget, val computeMs: Double)
@Serializable data class WorkerEndpoint(val id: String, val origin: String, val tokenFile: String)
@Serializable data class DistributedSettings(val format: String = "hyperl-distributed/1", val workers: List<WorkerEndpoint>,
    val maxParallelism: Int = 4, val shardElements: Int? = null, val coordinatorBudgetBytes: Long = 128L * 1024 * 1024,
    val requestTimeoutMs: Long = 10000, val jobTimeoutMs: Long = 60000) {
    fun validate() {
        require(format == "hyperl-distributed/1" && workers.size in 1..ShardProtocol.MAX_WORKERS)
        require(workers.map { it.id }.distinct().size == workers.size && workers.map { ShardProtocol.origin(it.origin).toString().trimEnd('/') }.distinct().size == workers.size)
        workers.forEach { require(it.id.matches(Regex("[A-Za-z][A-Za-z0-9_-]{0,63}")) && Path.of(it.tokenFile).isAbsolute); ShardProtocol.origin(it.origin) }
        require(maxParallelism in 1..ShardProtocol.MAX_WORKERS && (shardElements == null || shardElements in 1..HyperLContract.MAX_VECTOR_ELEMENTS))
        require(coordinatorBudgetBytes in 4L * 1024 * 1024..1024L * 1024 * 1024)
        require(requestTimeoutMs in 100..30000 && jobTimeoutMs in requestTimeoutMs..300000)
    }
}
