package org.hyperl
import kotlinx.serialization.Serializable
import java.net.URI

@Serializable data class ClusterProfile(val format:String="hyperl-cluster/1",val maxManagedNodes:Int=10000,
    val maxActiveWorkers:Int=8,val agentMayManage:Boolean=false,val agentNodeLimit:Int=64,
    val endpoints:List<String> = emptyList())
object ClusterPlanner {
    fun plan(profile:ClusterProfile):String {
        require(profile.format=="hyperl-cluster/1" && profile.maxManagedNodes in 1..1000000)
        require(profile.maxActiveWorkers in 1..256 && profile.maxActiveWorkers<=profile.maxManagedNodes)
        require(profile.agentNodeLimit in 0..profile.maxManagedNodes)
        require(profile.endpoints.size<=minOf(10000,profile.maxManagedNodes) && profile.endpoints.distinct().size==profile.endpoints.size)
        profile.endpoints.forEach{endpoint->
            require(endpoint.length<=2048)
            val uri=URI(endpoint);require(uri.scheme=="https" && uri.host!=null && uri.rawUserInfo==null && uri.rawFragment==null && uri.rawQuery==null && '%' !in uri.host && (uri.port==-1 || uri.port in 1..65535))
        }
        return "Validated inventory ceiling=${profile.maxManagedNodes}, worker ceiling=${profile.maxActiveWorkers}, " +
            "agentManagement=${profile.agentMayManage}, agentLimit=${profile.agentNodeLimit}. IPv6 literal/IPv4/DNS HTTPS endpoints accepted. " +
            "No discovery, enrollment, remote execution, automatic scale or simultaneous-node capacity is proven."
    }
}
