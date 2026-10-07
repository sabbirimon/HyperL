package org.hyperl

import kotlinx.coroutines.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.attribute.PosixFilePermissions
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs

/** Explicit local GPU bridge. Never silently falls back or assumes discovery is qualification. */
class OpenClBridge(private val executable:Path) {
    companion object { val cleanupUncertain=AtomicBoolean(false) }
    init {require(!cleanupUncertain.get()){"Previous native cleanup uncertain; GPU work remains blocked"};require(executable.isAbsolute && Files.isRegularFile(executable) && Files.isExecutable(executable)){"Supply an absolute installed OpenCL bridge executable path"}}
    suspend fun probe():String=withContext(Dispatchers.IO){
        privateDirectory { dir -> invoke(listOf("--probe"),dir);val file=dir.resolve("stdout.txt")
            require(Files.size(file)<=65536);Files.readString(file) }
    }
    suspend fun executeVerified(device:Int,program:HyperLProgram,inputs:Map<String,FloatArray>):FloatArray=withContext(Dispatchers.IO){
        require(device in 0..127);program.validate()
        require(program.instructions.none{it.operation=="sum"}){"OpenCL reductions not implemented"}
        // Current adapter verifies every request against reference semantics, not a speed benchmark.
        val expected=HyperLCpuBackend().execute(program,inputs)
        val n=inputs.values.first().size
        require(inputs.values.all{it.size==n}){"GPU elementwise inputs must have equal length"}
        privateDirectory{dir->
            Files.writeString(dir.resolve("kernel.cl"),"#pragma OPENCL FP_CONTRACT OFF\n"+HyperLSourceEmitter.emit(program,HyperLTarget.OPENCL_SPIRV).source)
            program.inputs.sorted().forEachIndexed{index,name->
                val bytes=ByteBuffer.allocate(n*4).order(ByteOrder.LITTLE_ENDIAN)
                inputs.getValue(name).forEach{bytes.putFloat(it)};Files.write(dir.resolve("input_$index.bin"),bytes.array())
            }
            invoke(listOf("--run",device.toString(),n.toString(),program.inputs.size.toString(),dir.toString()),dir)
            val output=dir.resolve("result.bin");require(Files.size(output)==n.toLong()*4){"Invalid GPU output length"}
            val bytes=ByteBuffer.wrap(Files.readAllBytes(output)).order(ByteOrder.LITTLE_ENDIAN)
            FloatArray(n){i->bytes.float.also{v->
                require(v.isFinite()){"GPU nonfinite intermediate/result"}
                require(abs(v.toDouble()-expected[i])<=1e-6*maxOf(1.0,abs(expected[i].toDouble()))){"GPU reference mismatch at $i"}
            }}
        }
    }
    private suspend fun invoke(arguments:List<String>,directory:Path) {
        currentCoroutineContext().ensureActive()
        val process=ProcessBuilder(listOf(executable.toString())+arguments).directory(directory.toFile())
            .redirectOutput(directory.resolve("stdout.txt").toFile()).redirectError(directory.resolve("stderr.txt").toFile()).start()
        try {
            withTimeout(20000){while(!process.waitFor(50,TimeUnit.MILLISECONDS)){currentCoroutineContext().ensureActive()}}
            require(process.exitValue()==0){"OpenCL bridge failed (exit ${process.exitValue()}); GPU/runtime may be unavailable"}
            require(Files.size(directory.resolve("stdout.txt"))<=65536 && Files.size(directory.resolve("stderr.txt"))<=65536){"Bridge output limit"}
        }finally{
            if(process.isAlive){process.destroyForcibly();process.waitFor(2,TimeUnit.SECONDS)}
            if(process.isAlive){cleanupUncertain.set(true);error("Native cleanup uncertain; GPU work remains blocked")}
        }
    }
    private suspend fun <T> privateDirectory(action:suspend (Path)->T):T {
        val directory=Files.createTempDirectory("hyperl-")
        try {if(Files.getFileStore(directory).supportsFileAttributeView("posix"))Files.setPosixFilePermissions(directory,PosixFilePermissions.fromString("rwx------"));return action(directory)}
        finally{Files.walk(directory).use{stream->stream.sorted(Comparator.reverseOrder()).forEach{Files.deleteIfExists(it)}}}
    }
}
