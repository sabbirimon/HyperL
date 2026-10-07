package org.hyperl
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.serialization.Serializable

/** HyperL v1: bounded declarative f32 vector kernels. No shell, pointers, host I/O,
 * recursive calls, arbitrary code or automatic backend substitution. */
@Serializable data class HyperLInstruction(val output:String,val operation:String,val inputs:List<String>)
@Serializable data class HyperLProgram(val format:String="hyperl/1",val inputs:Set<String>,val instructions:List<HyperLInstruction>,val output:String) {
    fun validate(){
        require(format=="hyperl/1") {"Unsupported HyperL language version"}
        require(inputs.size in 1..8 && instructions.size in 1..64)
        val defined=inputs.toMutableSet();require(defined.all{it.matches(NAME)})
        instructions.forEach { step ->
            require(step.output.matches(NAME) && step.output !in defined && step.inputs.all{it in defined}) {"Invalid HyperL dependency or duplicate output"}
            require(step.inputs.size == when(step.operation){"add","multiply"->2;"relu","sum"->1;else->error("Unsupported HyperL operation")})
            defined.add(step.output)
        }
        require(output in defined)
    }
}
@Serializable enum class HyperLTarget { CPU_REFERENCE, LLVM_CPU, METAL, CUDA, ROCM_HIP, VULKAN_SPIRV, OPENCL_SPIRV, SYCL, NPU_STABLEHLO, FPGA_VENDOR }
interface HyperLBackend {
    val target:HyperLTarget
    val runtimeRevision:String
    suspend fun execute(program:HyperLProgram,inputs:Map<String,FloatArray>):FloatArray
}
/** Genuine Kotlin/Android/JVM CPU backend. It is a correctness reference, not an LLM
 * inference engine or a CUDA-compatible compiler. Accelerated targets require adapters. */
class HyperLCpuBackend(private val memoryBudgetBytes:Long=MemoryPlanner.DEFAULT_BUDGET):HyperLBackend {
    override val target=HyperLTarget.CPU_REFERENCE
    override val runtimeRevision="hyperl-cpu/1"
    override suspend fun execute(program:HyperLProgram,inputs:Map<String,FloatArray>):FloatArray {
        program.validate();require(inputs.keys==program.inputs)
        require(inputs.values.all{it.size in 1..262144 && it.all(Float::isFinite)})
        var retained=inputs.values.sumOf{it.size.toLong()}
        require(retained<=1048576){"HyperL input vectors exceed retained memory budget"}
        MemoryPlanner.requireAdmission(MemoryPlanner.plan(program,inputs,memoryBudgetBytes))
        val live=inputs.mapValues{it.value.copyOf()}.toMutableMap()
        for(step in program.instructions){
            currentCoroutineContext().ensureActive()
            val args=step.inputs.map{live.getValue(it)};val length=if(step.operation=="sum") 1 else args[0].size
            require(step.operation !in setOf("add","multiply") || args[0].size==args[1].size){"HyperL shape mismatch; no implicit broadcasting"}
            retained+=length;require(retained<=1048576){"HyperL working memory exceeds 4 MiB vector budget"}
            val result=FloatArray(length)
            if(step.operation=="sum"){
                var sum=0f;args[0].forEachIndexed{i,value->if(i%1024==0) currentCoroutineContext().ensureActive();sum+=value};result[0]=sum
            }else for(i in 0 until length){
                if(i%1024==0) currentCoroutineContext().ensureActive()
                result[i]=when(step.operation){"add"->args[0][i]+args[1][i];"multiply"->args[0][i]*args[1][i];"relu"->maxOf(0f,args[0][i]);else->error("Unsupported operation")}
            }
            require(result.all(Float::isFinite)){"HyperL nonfinite result"};live[step.output]=result
        }
        return live.getValue(program.output).copyOf()
    }
}
/** Only explicitly registered backends execute. A target name never creates a driver. */
class HyperLRuntime(backends:List<HyperLBackend> = listOf(HyperLCpuBackend())) {
    private val adapters=backends.associateBy{it.target}.also{require(it.size==backends.size)}
    fun installedTargets()=adapters.keys
    suspend fun execute(target:HyperLTarget,program:HyperLProgram,inputs:Map<String,FloatArray>):FloatArray =
        (adapters[target] ?: error("HyperL backend unavailable: $target")).execute(program,inputs)
}
private val NAME=Regex("[A-Za-z][A-Za-z0-9_]{0,31}")
