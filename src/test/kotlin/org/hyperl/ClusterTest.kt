package org.hyperl
import org.junit.Assert.*
import org.junit.Test
class ClusterTest {
    @Test fun ipv6AcceptedWithoutNetworkOrCapacityClaim(){
        val plan=ClusterPlanner.plan(ClusterProfile(maxManagedNodes=1000000,endpoints=listOf("https://[2001:db8::1]:8443/node")))
        assertTrue(plan.contains("1000000"));assertTrue(plan.contains("No discovery"))
    }
    @Test fun rejectPlaintextCredentialsInvalidLimitsAndScopedIpv6(){
        for(url in listOf("http://[::1]/node","https://user:pass@[::1]/node","https://[fe80::1%25en0]/node"))assertThrows(IllegalArgumentException::class.java){ClusterPlanner.plan(ClusterProfile(endpoints=listOf(url)))}
        assertThrows(IllegalArgumentException::class.java){ClusterPlanner.plan(ClusterProfile(maxManagedNodes=1000001))}
        assertThrows(IllegalArgumentException::class.java){ClusterPlanner.plan(ClusterProfile(maxManagedNodes=1,maxActiveWorkers=8))}
    }
}
