package org.hyperl

import kotlinx.coroutines.*
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.attribute.PosixFilePermissions
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs

@Serializable data class MetalMetrics(val format:String,val device:String,val storageMode:String,val gpuCompletionConfirmed:Boolean,val compileMs:Double,val submitAndWaitMs:Double,val gpuMs:Double?=null)
// Required wire fields: the default JSON encoder omits properties with default values.
@Serializable data class MetalResult(val backend:String,val result:FloatArray,val device:String,val cpuVerified:Boolean,val metrics:MetalMetrics,val scope:String="This bounded elementwise request only; verification/copy/compile costs included, no speed claim")

/** Explicit reviewed Metal executable. Device discovery never qualifies execution. */
class MetalBridge(private val executable:Path){
    companion object {val cleanupUncertain=AtomicBoolean(false)}
    init {require(!cleanupUncertain.get()){"Metal completion unconfirmed; new Metal work is blocked"};require(executable.isAbsolute && Files.isRegularFile(executable) && Files.isExecutable(executable)){"Supply an absolute reviewed Metal bridge executable path"}}
    suspend fun probe():String=withContext(Dispatchers.IO){privateDirectory{dir->invoke(listOf("--probe"),dir);boundedText(dir.resolve("stdout.txt"))}}
    suspend fun executeVerified(index:Int,program:HyperLProgram,inputs:Map<String,FloatArray>):MetalResult {
        require(index in 0..127)
        val root=Workspace.json.parseToJsonElement(probe()).jsonObject
        require(root.getValue("format").jsonPrimitive.content=="hyperl-metal-probe/1")
        val devices=root.getValue("devices").jsonArray;require(devices.size<=128 && index in devices.indices){"Selected Metal GPU unavailable; no fallback"}
        val device=devices[index].jsonObject;require(device.getValue("index").jsonPrimitive.int==index)
        return executeVerified(device.getValue("name").jsonPrimitive.content,program,inputs)
    }
    suspend fun executeVerified(name:String,program:HyperLProgram,inputs:Map<String,FloatArray>):MetalResult=withContext(Dispatchers.IO){
        require(name.length in 1..256 && name.none{it.isISOControl()}){"Exact bounded Metal device name required"}
        program.validate();require(program.instructions.none{it.operation=="sum"}){"Metal reductions not implemented"}
        // Check every intermediate on CPU before allowing the generated GPU request.
        val expected=HyperLCpuBackend().execute(program,inputs);val n=inputs.values.first().size
        require(inputs.values.all{it.size==n}){"Metal elementwise inputs must have equal length"}
        privateDirectory{dir->
            Files.writeString(dir.resolve("kernel.metal"),PortableEmitter.emit(program,HyperLTarget.METAL).source)
            program.inputs.sorted().forEachIndexed{i,key->
                val bytes=ByteBuffer.allocate(n*4).order(ByteOrder.LITTLE_ENDIAN);inputs.getValue(key).forEach{bytes.putFloat(it)};Files.write(dir.resolve("input_$i.bin"),bytes.array())
            }
            invoke(listOf("--run",name,n.toString(),program.inputs.size.toString(),dir.toString()),dir)
            val metadata=Workspace.json.decodeFromString<MetalMetrics>(boundedText(dir.resolve("stdout.txt")))
            require(metadata.format=="hyperl-metal-result/1" && metadata.device==name && metadata.gpuCompletionConfirmed && metadata.storageMode in setOf("shared","managed")){"Metal device/completion metadata mismatch"}
            require(listOfNotNull(metadata.compileMs,metadata.submitAndWaitMs,metadata.gpuMs).all{it.isFinite() && it in 0.0..20000.0}){"Invalid Metal timing metadata"}
            val file=dir.resolve("result.bin");require(Files.size(file)==n.toLong()*4){"Invalid Metal result length"}
            val bytes=ByteBuffer.wrap(Files.readAllBytes(file)).order(ByteOrder.LITTLE_ENDIAN)
            val result=FloatArray(n){i->bytes.float.also{v->require(v.isFinite() && abs(v.toDouble()-expected[i])<=1e-6*maxOf(1.0,abs(expected[i].toDouble()))){"Metal CPU-reference mismatch at $i"}}}
            MetalResult(backend="METAL_GPU",result=result,device=name,cpuVerified=true,metrics=metadata)
        }
    }
    private fun boundedText(file:Path):String {require(Files.size(file)<=65536){"Metal output limit"};return Files.readString(file)}
    private suspend fun invoke(arguments:List<String>,dir:Path){
        currentCoroutineContext().ensureActive();require(!cleanupUncertain.get()){"Metal completion unconfirmed; new work blocked"}
        val process=ProcessBuilder(listOf(executable.toString())+arguments).directory(dir.toFile()).redirectOutput(dir.resolve("stdout.txt").toFile()).redirectError(dir.resolve("stderr.txt").toFile()).start()
        try {
            withTimeout(20000){while(!process.waitFor(50,TimeUnit.MILLISECONDS)){currentCoroutineContext().ensureActive()}}
            require(Files.size(dir.resolve("stdout.txt"))<=65536 && Files.size(dir.resolve("stderr.txt"))<=65536){"Metal output limit"}
            require(process.exitValue()==0){"Metal bridge failed (exit ${process.exitValue()}): ${boundedText(dir.resolve("stderr.txt")).take(2048)}"}
        }finally{
            if(process.isAlive){process.destroyForcibly();process.waitFor(2,TimeUnit.SECONDS)}
            if(process.isAlive || Files.exists(dir.resolve("submitted.marker"))){cleanupUncertain.set(true);error("Metal GPU completion unconfirmed; further Metal work blocked. Killing the host does not prove GPU cancellation.")}
        }
    }
    private suspend fun <T> privateDirectory(action:suspend(Path)->T):T {
        val dir=Files.createTempDirectory("hyperl-metal-")
        try {if(Files.getFileStore(dir).supportsFileAttributeView("posix"))Files.setPosixFilePermissions(dir,PosixFilePermissions.fromString("rwx------"));return action(dir)}
        finally{Files.walk(dir).use{stream->stream.sorted(Comparator.reverseOrder()).forEach{Files.deleteIfExists(it)}}}
    }
}
