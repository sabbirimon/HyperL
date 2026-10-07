package org.hyperl
import kotlinx.serialization.Serializable

@Serializable enum class TelecomGeneration { LTE_4G, NR_5G, RESEARCH_6G }
@Serializable data class TelecomProfile(val format:String="hyperl-telecom/1",val generation:TelecomGeneration,
    val standardReferences:List<String>,val labOnly:Boolean=true,val ownerApproved:Boolean=false,
    val isolatedLabReady:Boolean=false,val adapterInstalled:Boolean=false,val conformanceEvidenceSha256:String?=null)
object TelecomPlanner {
    fun plan(profile:TelecomProfile):String {
        require(profile.format=="hyperl-telecom/1")
        require(profile.standardReferences.size in 1..32 && profile.standardReferences.all{it.length in 1..256})
        require(profile.conformanceEvidenceSha256==null || profile.conformanceEvidenceSha256.matches(Regex("[a-f0-9]{64}")))
        // A profile and an unverified hash grant no access, certification or transmitter capability.
        return "${profile.generation}: research plan only; network stack and radio adapter unavailable. " +
            "Execution disabled, certified=false. Requires independently verified standard editions, " +
            "owner scope, isolated lab, adapter, authentication, conformance and radio authorization."
    }
}
