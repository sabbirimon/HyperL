package org.hyperl

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.*
import java.nio.file.Files
import java.nio.file.Path

object Workspace {
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
        return json.decodeFromString<HyperLProgram>(text).also{it.validate()}
    }
    fun inputs(text:String):Map<String,FloatArray> {
        require(text.length<=MAX_JSON_BYTES)
        val root=json.parseToJsonElement(text).jsonObject
        require(root.size in 1..8)
        var total=0L
        return root.mapValues{(_,value)->
            val items=value.jsonArray
            require(items.size in 1..262144)
            total+=items.size;require(total<=1048576){"Input exceeds retained vector budget"}
            FloatArray(items.size){index->
                val number=items[index].jsonPrimitive
                require(!number.isString){"Numbers must not be strings"}
                number.float.also{require(it.isFinite()){"Nonfinite input"}}
            }
        }
    }
    fun result(values:FloatArray):String=json.encodeToString(JsonArray(values.map{JsonPrimitive(it)}))
    fun capabilities():String=buildJsonObject {
        put("version","0.1.0-alpha.1");put("language","hyperl/1")
        put("os",System.getProperty("os.name"));put("architecture",System.getProperty("os.arch"))
        put("java",System.getProperty("java.version"))
        put("cpuReferenceAvailable",true);put("automaticGpuFallback",false)
        put("nativeBridgeRequiresExplicitPath",true);put("telecomCertified",false)
        put("iosPackageAvailable",false)
        put("memoryAwareCpuAdmission",true)
        put("memory",json.encodeToJsonElement(MemoryPlanner.observe()))
    }.toString()
}

fun cpu(program:String,inputs:String,budgetBytes:Long=MemoryPlanner.DEFAULT_BUDGET):String=runBlocking {
    Workspace.result(HyperLCpuBackend(budgetBytes).execute(Workspace.program(program),Workspace.inputs(inputs)))
}
