package org.hyperl

import java.lang.management.ManagementFactory
import kotlinx.serialization.Serializable

/** Observed JVM/environment snapshot, never a reservation or accelerator topology guess. */
@Serializable data class MemorySnapshot(val heapLimitBytes:Long,val heapUsedBytes:Long,val environmentTotalBytes:Long?=null,val environmentFreeBytes:Long?=null) {
    fun availableBytes():Long {
        require(heapLimitBytes>0 && heapUsedBytes in 0..heapLimitBytes)
        val heap=(heapLimitBytes-heapUsedBytes-32L*1024*1024).coerceAtLeast(0)/2
        return heap // Free OS pages do not include every reclaimable cache; advisory only.
    }
}
@Serializable data class MemoryTier(val type:String,val available:Boolean?=null,val capacityBytes:Long?=null)
@Serializable data class MemoryPlan(val inputBytes:Long,val retainedVectorBytes:Long,val estimatedArrayAndWorkspaceBytes:Long,val selectedBudgetBytes:Long,val observedAdmissionBytes:Long,val admitted:Boolean,val reason:String,val snapshot:MemorySnapshot,val tiers:List<MemoryTier>,val environmentPressureAdvisory:Boolean?,val automaticSpill:Boolean=false,val reservationGuaranteed:Boolean=false)
object MemoryPlanner {
    const val DEFAULT_BUDGET=16L*1024*1024
    fun observe():MemorySnapshot {
        val runtime=Runtime.getRuntime()
        val bean=ManagementFactory.getOperatingSystemMXBean() as? com.sun.management.OperatingSystemMXBean
        fun observed(action:()->Long):Long?=runCatching(action).getOrNull()?.takeIf{it>=0}
        return MemorySnapshot(runtime.maxMemory(),(runtime.totalMemory()-runtime.freeMemory()).coerceIn(0,runtime.maxMemory()),bean?.let{observed{it.totalMemorySize}},bean?.let{observed{it.freeMemorySize}})
    }
    fun plan(program:HyperLProgram,inputs:Map<String,FloatArray>,budgetBytes:Long=DEFAULT_BUDGET,snapshot:MemorySnapshot=observe()):MemoryPlan {
        program.validate();require(inputs.keys==program.inputs);require(budgetBytes in 1..(1024L*1024*1024))
        require(inputs.values.all{it.size in 1..262144})
        val lengths=inputs.mapValues{it.value.size}.toMutableMap()
        val inputElements=inputs.values.sumOf{it.size.toLong()};var retained=inputElements
        program.instructions.forEach{step->
            val args=step.inputs.map{lengths.getValue(it)}
            require(step.operation !in setOf("add","multiply") || args[0]==args[1]){"HyperL shape mismatch; no implicit broadcasting"}
            val n=if(step.operation=="sum")1 else args[0];lengths[step.output]=n;retained+=n
        }
        val estimated=(inputElements+retained+lengths.getValue(program.output))*4+32L*(inputs.size*2+program.instructions.size+1)+65536
        val available=minOf(budgetBytes,snapshot.availableBytes())
        val reason=when {retained>1048576->"Retained vector format limit exceeded";estimated>available->"Selected budget or observed JVM heap headroom insufficient";else->"Admitted for bounded CPU arrays; memory is not reserved"}
        return MemoryPlan(inputElements*4,retained*4,estimated,budgetBytes,available,retained<=1048576 && estimated<=available,reason,snapshot,listOf(MemoryTier("JVM_HEAP",true,snapshot.heapLimitBytes),MemoryTier("SYSTEM_RAM",snapshot.environmentTotalBytes?.let{true},snapshot.environmentTotalBytes),MemoryTier("HBM"),MemoryTier("GDDR"),MemoryTier("DDR_LPDDR"),MemoryTier("UNIFIED_GPU"),MemoryTier("SRAM"),MemoryTier("CXL_NUMA"),MemoryTier("SSD_NVME")),snapshot.environmentFreeBytes?.let{it<estimated})
    }
    fun requireAdmission(plan:MemoryPlan){require(plan.admitted){"Memory admission rejected: ${plan.reason}; estimated=${plan.estimatedArrayAndWorkspaceBytes}, available=${plan.observedAdmissionBytes}"}}
    fun requireStreamingHeadroom(){require(observe().availableBytes()>=32L*1024*1024){"Insufficient observed memory headroom for bounded encrypted dataset buffers"}}
}
