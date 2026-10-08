package org.hyperl

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.*
import java.nio.file.Files
import java.nio.file.Path

object Workspace {
    const val VERSION="0.1.0-alpha.5"
    const val MAX_JSON_BYTES=16*1024*1024
    val json=Json { prettyPrint=true; ignoreUnknownKeys=false }
    const val EXAMPLE_PROGRAM="""{"format":"hyperl/1","inputs":["x","w"],"instructions":[{"output":"value","operation":"multiply","inputs":["x","w"]},{"output":"positive","operation":"relu","inputs":["value"]}],"output":"positive"}"""
    const val EXAMPLE_INPUTS="""{"x":[-1,2,3],"w":[2,3,4]}"""
    fun read(file:Path):String {
        require(Files.isRegularFile(file) && Files.size(file)<=MAX_JSON_BYTES){"Input must be a regular file of at most 16 MiB"}
        Files.newInputStream(file).use { input ->
            val bytes=input.readNBytes(MAX_JSON_BYTES+1)
            require(bytes.size<=MAX_JSON_BYTES){"Input exceeds 16 MiB"}
            return bytes.toString(Charsets.UTF_8)
        }
    }
    fun program(text:String):HyperLProgram {
        require(text.length<=MAX_JSON_BYTES)
        return HyperLCodec.program(text)
    }
    fun inputs(text:String):Map<String,FloatArray> {
        return HyperLCodec.inputs(text)
    }
    fun result(values:FloatArray):String=json.encodeToString(JsonArray(values.map{JsonPrimitive(it)}))
    fun capabilities():String=buildJsonObject {
        put("version",VERSION);put("language","hyperl/1")
        put("os",System.getProperty("os.name"));put("architecture",System.getProperty("os.arch"))
        put("java",System.getProperty("java.version"))
        put("cpuReferenceAvailable",true);put("automaticGpuFallback",false)
        put("nativeBridgeRequiresExplicitPath",true);put("telecomCertified",false)
        put("metalBridgeAvailableAsSource",true);put("metalHardwareQualified",false)
        put("iosPackageAvailable",false)
        put("memoryAwareCpuAdmission",true)
        put("memory",json.encodeToJsonElement(MemoryPlanner.observe()))
    }.toString()
}

fun cpu(program:String,inputs:String,budgetBytes:Long=MemoryPlanner.DEFAULT_BUDGET):String=runBlocking {
    Workspace.result(HyperLCpuBackend(budgetBytes).execute(Workspace.program(program),Workspace.inputs(inputs)))
}
