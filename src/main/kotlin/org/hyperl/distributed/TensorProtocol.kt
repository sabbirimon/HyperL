// SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
// Copyright 2026 Sabbir Hassan Imon and respective contributors.
package org.hyperl.distributed

import org.hyperl.*
import kotlinx.serialization.Serializable
import java.io.*
import java.nio.ByteBuffer

/** Contiguous, row-major f32 only. Shape is metadata; kernels use the flattened range. */
@Serializable data class TensorHandle(val id: String, val version: Long, val shape: List<Int>, val dtype: String = "f32") {
    fun elements(): Int {
        ShardProtocol.uuid(id); require(version in 1..1_000_000_000 && dtype == "f32" && shape.size in 1..8)
        var n = 1L
        shape.forEach { require(it in 1..HyperLContract.MAX_VECTOR_ELEMENTS); n *= it; require(n <= HyperLContract.MAX_VECTOR_ELEMENTS) }
        return n.toInt()
    }
}
@Serializable data class TensorSlice(val tensorId: String, val version: Long, val start: Int, val end: Int) {
    fun validate() { ShardProtocol.uuid(tensorId); require(version in 1..1_000_000_000 && start >= 0 && end > start && end <= HyperLContract.MAX_VECTOR_ELEMENTS) }
}
@Serializable data class TensorChunk(val slice: TensorSlice, val sha256: String) {
    fun validate() { slice.validate(); require(sha256.matches(Regex("[a-f0-9]{64}"))) }
}
@Serializable internal data class TensorRequest(val format: String = TensorProtocol.FORMAT, val jobId: String,
    val sessionId: String, val workerInstanceId: String, val action: String, val chunks: List<TensorChunk> = emptyList(),
    val inputs: Map<String, TensorChunk> = emptyMap(), val output: TensorSlice? = null, val program: HyperLProgram? = null,
    val timeoutMs: Long = 10000)
@Serializable internal data class TensorReply(val format: String = TensorProtocol.FORMAT, val jobId: String,
    val requestSha256: String, val chunks: List<TensorChunk>, val backend: HyperLTarget)

/** New envelope: the existing hyperl-shard/1 format and native ABI are unchanged. */
internal object TensorProtocol {
    const val FORMAT = "hyperl-tensor/1"
    const val CONTENT_TYPE = "application/vnd.hyperl.tensor-v1"
    const val LEASE_MS = 300_000L
    const val MAX_CHUNKS = 4096
    fun digest(values: FloatArray): String {
        val bytes = ByteBuffer.allocate(4 * values.size)
        values.forEach { require(it.isFinite()); bytes.putFloat(it) }
        return ShardProtocol.digest(bytes.array())
    }
    fun validate(request: TensorRequest) {
        require(request.format == FORMAT); ShardProtocol.uuid(request.jobId); ShardProtocol.uuid(request.sessionId); ShardProtocol.uuid(request.workerInstanceId)
        require(request.timeoutMs in 100..30000 && request.chunks.size <= 8 && request.inputs.size <= 8)
        request.chunks.forEach { it.validate() }; request.inputs.values.forEach { it.validate() }
        when (request.action) {
            "put", "get", "drop" -> require(request.chunks.isNotEmpty() && request.inputs.isEmpty() && request.output == null && request.program == null)
            "clear" -> require(request.chunks.isEmpty() && request.inputs.isEmpty() && request.output == null && request.program == null)
            "execute" -> {
                require(request.chunks.isEmpty()); val program = requireNotNull(request.program); program.validate()
                require(program.instructions.all { it.operation in ShardProtocol.operations }) { "Tensor engine supports elementwise add/multiply/relu only" }
                val output = requireNotNull(request.output); output.validate()
                require(request.inputs.keys == program.inputs && request.inputs.values.all { it.slice.start == output.start && it.slice.end == output.end })
                require((program.inputs.size + program.instructions.size).toLong() * (output.end - output.start) <= HyperLContract.MAX_RETAINED_ELEMENTS)
            }
            else -> error("Unsupported tensor action")
        }
    }
    fun request(meta: TensorRequest, values: List<FloatArray> = emptyList()): ByteArray {
        validate(meta); validateValues(if (meta.action == "put") meta.chunks else emptyList(), values)
        return frame(ShardProtocol.json.encodeToString(meta), values)
    }
    fun readRequest(bytes: ByteArray): Pair<TensorRequest, List<FloatArray>> = unframe(bytes) { text, input ->
        val meta = ShardProtocol.json.decodeFromString<TensorRequest>(text); validate(meta)
        meta to readValues(input, if (meta.action == "put") meta.chunks else emptyList())
    }
    fun reply(meta: TensorReply, values: List<FloatArray>): ByteArray {
        validateValues(meta.chunks, values); return frame(ShardProtocol.json.encodeToString(meta), values)
    }
    fun readReply(bytes: ByteArray): Pair<TensorReply, List<FloatArray>> = unframe(bytes) { text, input ->
        val meta = ShardProtocol.json.decodeFromString<TensorReply>(text)
        require(meta.format == FORMAT && meta.chunks.size <= 8); ShardProtocol.uuid(meta.jobId)
        require(meta.requestSha256.matches(Regex("[a-f0-9]{64}")))
        meta to readValues(input, meta.chunks)
    }
    private fun validateValues(chunks: List<TensorChunk>, values: List<FloatArray>) {
        require(chunks.size == values.size)
        chunks.zip(values).forEach { (chunk, data) -> chunk.validate(); require(data.size == chunk.slice.end - chunk.slice.start && digest(data) == chunk.sha256) }
    }
    private fun readValues(input: DataInputStream, chunks: List<TensorChunk>): List<FloatArray> {
        chunks.forEach { it.validate() }
        require(input.available().toLong() == chunks.sumOf { 4L * (it.slice.end - it.slice.start) })
        return chunks.map { chunk -> FloatArray(chunk.slice.end - chunk.slice.start) { input.readFloat().also { require(it.isFinite()) } }
            .also { require(digest(it) == chunk.sha256) { "Tensor chunk digest mismatch" } } }
    }
    private fun frame(text: String, values: List<FloatArray>): ByteArray {
        val metadata = text.toByteArray(Charsets.UTF_8); require(metadata.size in 1..ShardProtocol.MAX_METADATA)
        val size = 8L + metadata.size + values.sumOf { 4L * it.size }; require(size <= ShardProtocol.MAX_FRAME)
        val bytes = ByteArrayOutputStream(size.toInt())
        DataOutputStream(bytes).use { out -> out.writeInt(0x484c5431); out.writeInt(metadata.size); out.write(metadata); values.forEach { it.forEach(out::writeFloat) } }
        return bytes.toByteArray()
    }
    private fun <T> unframe(bytes: ByteArray, body: (String, DataInputStream) -> T): T {
        require(bytes.size in 8..ShardProtocol.MAX_FRAME)
        return DataInputStream(ByteArrayInputStream(bytes)).use { input ->
            require(input.readInt() == 0x484c5431); val size = input.readInt()
            require(size in 1..ShardProtocol.MAX_METADATA && size <= input.available())
            val text = input.readNBytes(size).toString(Charsets.UTF_8); ShardProtocol.checkJson(text)
            body(text, input).also { require(input.available() == 0) }
        }
    }
}

/** Authoritative worker payloads share the worker's credit pool; only explicit drop/expiry evicts them. */
internal class TensorStore(private val credits: MemoryCredits, private val now: () -> Long = System::nanoTime) {
    private data class Key(val session: String, val slice: TensorSlice)
    private data class Entry(val chunk: TensorChunk, val data: FloatArray, var expires: Long)
    private val entries = mutableMapOf<Key, Entry>()
    private var closed = false
    @Synchronized fun expire() { val time = now(); entries.entries.removeIf { (_, entry) ->
        if (time - entry.expires >= 0) { credits.release(cost(entry.data)); true } else false } }
    @Synchronized fun put(session: String, chunks: List<TensorChunk>, values: List<FloatArray>) {
        check(!closed); expire(); require(chunks.size == values.size && chunks.map { it.slice }.distinct().size == chunks.size)
        chunks.zip(values).forEach { (chunk, data) -> chunk.validate(); require(chunk.slice.end - chunk.slice.start == data.size && TensorProtocol.digest(data) == chunk.sha256)
            entries[Key(session, chunk.slice)]?.let { require(it.chunk == chunk) { "Immutable tensor version conflict" } } }
        val new = chunks.zip(values).filter { Key(session, it.first.slice) !in entries }
        require(entries.size + new.size <= TensorProtocol.MAX_CHUNKS) { "Worker tensor directory full" }
        val bytes = new.sumOf { cost(it.second) }; check(credits.acquire(bytes)) { "Worker tensor memory exhausted" }
        try {
            val snapshots = new.map { (chunk, data) -> Key(session, chunk.slice) to Entry(chunk, data.copyOf(), deadline()) }
            snapshots.forEach { (key, entry) -> entries[key] = entry }
        } catch (e: Throwable) { new.forEach { entries.remove(Key(session, it.first.slice)) }; credits.release(bytes); throw e }
        chunks.forEach { entries.getValue(Key(session, it.slice)).expires = deadline() }
    }
    @Synchronized fun get(session: String, chunks: List<TensorChunk>): List<FloatArray> {
        expire()
        return chunks.map { chunk -> val entry = entries[Key(session, chunk.slice)] ?: error("Tensor lease expired or chunk unavailable")
            require(entry.chunk == chunk) { "Tensor version/digest conflict" }; entry.expires = deadline(); entry.data.copyOf() }
    }
    @Synchronized fun drop(session: String, chunks: List<TensorChunk>) { chunks.forEach { chunk ->
        val key = Key(session, chunk.slice); entries[key]?.let { require(it.chunk == chunk); entries.remove(key); credits.release(cost(it.data)) } } }
    @Synchronized fun clear(session: String) { entries.entries.removeIf { (key, entry) -> if (key.session == session) { credits.release(cost(entry.data)); true } else false } }
    @Synchronized fun close() { closed = true; entries.values.forEach { credits.release(cost(it.data)) }; entries.clear() }
    private fun deadline() = now() + TensorProtocol.LEASE_MS * 1_000_000
    private fun cost(values: FloatArray) = 4L * values.size + 512 // payload plus bounded entry accounting
}
