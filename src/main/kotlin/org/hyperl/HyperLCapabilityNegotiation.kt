// SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
// Copyright 2026 Sabbir Hassan Imon and respective contributors.
package org.hyperl

/** Classification is inventory only: a bus/NIC/vendor label never enables execution. */
enum class HyperLFabricKind { PCIE, CXL, NVLINK, INFINITY_FABRIC, ETHERNET, INFINIBAND, ROCE, USB, CUSTOM }
enum class HyperLFabricScope { HOST_IO, COHERENT_MEMORY, ACCELERATOR_SCALE_UP, NETWORK, EXTENSION }
enum class HyperLHardwareClass { CPU, GPU, NPU, FPGA, ASIC, OTHER }
data class HyperLFabricProfile(val kind: HyperLFabricKind, val scope: HyperLFabricScope,
    val qualificationRequirements: List<String>, val extensionId: String?, val executionProvided: Boolean)

data class HyperLWireContract(val protocolId: String, val version: Int)

/** Exact workload semantics; shape/dtype/operator coverage must already be qualified.
 * programSha256 identifies common logical work, not a vendor-specific executable.
 */
data class HyperLWorkloadContract(val programSha256: String, val shapeSignature: String,
    val dtype: String, val numericalMode: String)
data class HyperLFeatureContract(val featureId: String, val protocol: HyperLWireContract,
    val workload: HyperLWorkloadContract)

/** This tuple is one qualification scope, avoiding a Cartesian product of unrelated
 * protocol versions, shapes, dtypes and numerical modes. The adapter's artifact hash
 * identifies executable provenance; FPGA/ASIC configuration identifies the separately
 * qualified bitstream/firmware. These are caller-supplied evidence, not attestations.
 */
data class HyperLFeatureOffer(val contract: HyperLFeatureContract,
    val adapter: HyperLAdapterDescriptor, val hardwareClass: HyperLHardwareClass,
    val supportedOperations: Set<String>, val driverRevision: String, val observedAtMs: Long,
    val ownerEnabled: Boolean = false, val qualified: Boolean = false,
    val firmwareRevision: String? = null, val configurationSha256: String? = null)

data class HyperLNodeCapabilities(val deviceId: String, val os: String, val abi: String,
    val observedAtMs: Long, val offers: List<HyperLFeatureOffer>,
    val ownerApproved: Boolean = false, val identityVerified: Boolean = false)

/** One direction of a specifically identified lane. Both directions of the same lane
 * must agree on kind, extension and wire contract. Authentication is transport-specific
 * evidence, not a claim that raw PCIe/CXL/NVLink supplies an authenticated network.
 */
data class HyperLFabricLane(val laneId: String, val fromDeviceId: String, val toDeviceId: String,
    val kind: HyperLFabricKind, val protocol: HyperLWireContract, val adapterId: String,
    val adapterRevision: String, val observedAtMs: Long, val extensionId: String? = null,
    val ownerApproved: Boolean = false, val qualified: Boolean = false, val authenticated: Boolean = false)

/** Supplied by the local implementation registry, never directly from peer advertisements.
 * A requested contract says the caller has an implementation; negotiation does not create it.
 */
data class HyperLRequestedCapability(val contract: HyperLFeatureContract, val requiredOperations: Set<String>)
/** Qualification snapshots must be rebound to current installed artifacts and lane
 * revisions at dispatch. IDs alone are not a stable qualification identity.
 */
data class HyperLNegotiatedCapability(val contract: HyperLFeatureContract, val commonOperations: Set<String>,
    val leftOffer: HyperLFeatureOffer, val rightOffer: HyperLFeatureOffer,
    val eligibleLanes: List<HyperLFabricLane>) {
    val leftAdapterId: String get() = leftOffer.adapter.id
    val rightAdapterId: String get() = rightOffer.adapter.id
    val eligibleLaneIds: List<String> get() = eligibleLanes.map { it.laneId }.distinct().sorted()
}
data class HyperLCapabilityDecision(val contract: HyperLFeatureContract,
    val candidate: HyperLNegotiatedCapability?, val reason: String)
data class HyperLCapabilityAgreement(val leftDeviceId: String, val rightDeviceId: String,
    val decisions: List<HyperLCapabilityDecision>, val executionActivated: Boolean,
    val resourceReservationGuaranteed: Boolean)

/** Pure, bounded pair negotiation. Call per coordinator/worker pair before sharding
 * admission, and recheck at dispatch. Never sums memory or infers capabilities from
 * device price/model, common vendor, processor class or physical fabric classification.
 * No discovery, trust enrollment, adapter loading, distributed consensus or execution.
 */
object HyperLCapabilityNegotiation {
    const val MAX_AGE_MS = 30_000L
    private val id = Regex("[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,127}")
    private val sha = Regex("[a-f0-9]{64}")

    fun fabricProfile(kind: HyperLFabricKind, extensionId: String? = null): HyperLFabricProfile {
        if (kind == HyperLFabricKind.CUSTOM) require(extensionId != null && id.matches(extensionId)) { "Custom fabric needs an explicit extension identity" }
        else require(extensionId == null) { "Extension identity belongs to a custom fabric" }
        val (scope, requirements) = when (kind) {
            HyperLFabricKind.PCIE -> HyperLFabricScope.HOST_IO to listOf("Exact endpoint/root-complex topology", "Qualified transfer adapter and buffer ownership", "Explicit DMA isolation and peer-access policy")
            HyperLFabricKind.CXL -> HyperLFabricScope.COHERENT_MEMORY to listOf("Exact CXL protocol/version and topology", "Qualified memory/coherency adapter", "Physical memory ownership and alias accounting")
            HyperLFabricKind.NVLINK -> HyperLFabricScope.ACCELERATOR_SCALE_UP to listOf("Exact supported NVLink/NVSwitch topology", "Qualified vendor runtime and peer-memory/collective operation", "Buffer lifetime and isolation; not inferred from NIC presence")
            HyperLFabricKind.INFINITY_FABRIC -> HyperLFabricScope.ACCELERATOR_SCALE_UP to listOf("Exact supported AMD fabric topology", "Qualified runtime and transfer/collective operation", "Explicit memory ownership and isolation")
            HyperLFabricKind.ETHERNET -> HyperLFabricScope.NETWORK to listOf("Observed NIC/link and selected protocol", "Qualified transport adapter with authenticated peer scope", "Measured loss/backpressure and buffer ownership")
            HyperLFabricKind.INFINIBAND -> HyperLFabricScope.NETWORK to listOf("Qualified HCA, driver and fabric configuration", "Exact verbs/UCX or other installed adapter protocol", "Registered-memory lifetime, access scope and cancellation")
            HyperLFabricKind.ROCE -> HyperLFabricScope.NETWORK to listOf("Qualified RoCE NIC/driver and Ethernet fabric configuration", "Exact installed transport/version", "Registered-memory access, congestion behavior and cancellation")
            HyperLFabricKind.USB -> HyperLFabricScope.HOST_IO to listOf("Observed USB transport and endpoint roles", "Qualified transfer or owner-configured tunnel adapter", "Authenticated peer and bounded transfer lifetime")
            HyperLFabricKind.CUSTOM -> HyperLFabricScope.EXTENSION to listOf("Versioned extension specification", "Independently qualified installed adapter", "Explicit topology, ownership, authentication and cancellation contracts")
        }
        return HyperLFabricProfile(kind, scope, requirements, extensionId, false)
    }

    fun negotiate(left: HyperLNodeCapabilities, right: HyperLNodeCapabilities,
        lanes: List<HyperLFabricLane>, implementedRequests: List<HyperLRequestedCapability>, nowMs: Long): HyperLCapabilityAgreement {
        require(nowMs > 0 && left.deviceId != right.deviceId)
        validate(left); validate(right)
        require(lanes.size <= 32 && implementedRequests.size in 1..16)
        require(implementedRequests.map { it.contract }.distinct().size == implementedRequests.size) { "Duplicate local capability contract" }
        implementedRequests.forEach { validate(it.contract); operations(it.requiredOperations) }
        require(lanes.map { it.laneId to it.fromDeviceId }.distinct().size == lanes.size) { "Duplicate directed lane" }
        lanes.forEach {
            require(id.matches(it.laneId) && id.matches(it.adapterId) && label(it.adapterRevision))
            require(setOf(it.fromDeviceId, it.toDeviceId) == setOf(left.deviceId, right.deviceId)) { "Lane must bind these two distinct devices" }
            validate(it.protocol); fabricProfile(it.kind, it.extensionId)
        }
        val nodeReady = listOf(left, right).all { it.ownerApproved && it.identityVerified && fresh(it.observedAtMs, nowMs) }
        val decisions = implementedRequests.sortedBy { sortKey(it.contract) }.map { request ->
            val contract = request.contract
            fun blocked(reason: String) = HyperLCapabilityDecision(contract, null, reason)
            if (!nodeReady) return@map blocked("Both identities need current verification and owner approval")
            val leftOffers = left.offers.filter { it.contract == contract && eligible(it, left, nowMs) }.sortedBy { it.adapter.id }
            val rightOffers = right.offers.filter { it.contract == contract && eligible(it, right, nowMs) }.sortedBy { it.adapter.id }
            if (leftOffers.isEmpty() || rightOffers.isEmpty()) return@map blocked("No fresh qualified offer for this exact feature/protocol/version/workload on both devices")
            val pairedLaneIds = lanes.filter { it.fromDeviceId == left.deviceId && it.protocol == contract.protocol && eligible(it, nowMs) }
                .filter { outbound -> lanes.any { inbound -> inbound.laneId == outbound.laneId && inbound.fromDeviceId == right.deviceId &&
                    inbound.protocol == outbound.protocol && inbound.kind == outbound.kind && inbound.extensionId == outbound.extensionId && eligible(inbound, nowMs) } }
                .map { it.laneId }.sorted()
            if (pairedLaneIds.isEmpty()) return@map blocked("No fresh qualified authenticated bidirectional lane for this wire contract")
            val pairedLanes = lanes.filter { it.laneId in pairedLaneIds }
                .sortedWith(compareBy({ it.laneId }, { it.fromDeviceId }))
            var candidate: HyperLNegotiatedCapability? = null
            for (a in leftOffers) {
                for (b in rightOffers) {
                    val common = a.supportedOperations.intersect(b.supportedOperations).sorted().toSet()
                    if (common.containsAll(request.requiredOperations)) {
                        candidate = HyperLNegotiatedCapability(contract, common,
                            a.copy(supportedOperations = a.supportedOperations.toSet()),
                            b.copy(supportedOperations = b.supportedOperations.toSet()), pairedLanes)
                        break
                    }
                }
                if (candidate != null) break
            }
            if (candidate == null) blocked("Required operators are not jointly qualified for this exact contract")
            else HyperLCapabilityDecision(contract, candidate, "Eligible pair only; dispatch, resource admission and ongoing permission checks remain required")
        }
        return HyperLCapabilityAgreement(left.deviceId, right.deviceId, decisions, false, false)
    }

    private fun eligible(offer: HyperLFeatureOffer, node: HyperLNodeCapabilities, now: Long): Boolean =
        offer.ownerEnabled && offer.qualified && fresh(offer.observedAtMs, now) && offer.adapter.eligible(node.abi, node.os) &&
            when (offer.hardwareClass) {
                HyperLHardwareClass.FPGA -> offer.firmwareRevision != null && offer.configurationSha256 != null &&
                    offer.adapter.target == HyperLTarget.FPGA_VENDOR
                HyperLHardwareClass.ASIC -> offer.firmwareRevision != null && offer.configurationSha256 != null &&
                    offer.adapter.target !in setOf(HyperLTarget.CPU_REFERENCE, HyperLTarget.LLVM_CPU, HyperLTarget.FPGA_VENDOR)
                else -> true
            }

    private fun eligible(lane: HyperLFabricLane, now: Long) = lane.ownerApproved && lane.qualified && lane.authenticated && fresh(lane.observedAtMs, now)
    private fun fresh(observed: Long, now: Long) = observed > 0 && observed <= now && now - observed <= MAX_AGE_MS
    private fun label(value: String) = value.length in 1..128 && value.all { it.code in 0x21..0x7e }
    private fun operations(values: Set<String>) { require(values.size in 1..64 && values.all { id.matches(it) }) }
    private fun validate(protocol: HyperLWireContract) { require(id.matches(protocol.protocolId) && protocol.version in 1..65535) }
    private fun validate(contract: HyperLFeatureContract) {
        require(id.matches(contract.featureId)); validate(contract.protocol)
        require(sha.matches(contract.workload.programSha256) && label(contract.workload.shapeSignature) && label(contract.workload.dtype) && label(contract.workload.numericalMode))
    }
    private fun validate(node: HyperLNodeCapabilities) {
        require(id.matches(node.deviceId) && label(node.os) && label(node.abi) && node.offers.size <= 64)
        require(node.offers.map { it.contract to it.adapter.id }.distinct().size == node.offers.size) { "Duplicate adapter contract offer" }
        node.offers.forEach { offer ->
            validate(offer.contract); operations(offer.supportedOperations)
            require(label(offer.driverRevision))
            require(offer.firmwareRevision == null || label(offer.firmwareRevision))
            require(offer.configurationSha256 == null || sha.matches(offer.configurationSha256))
            val adapter = offer.adapter
            require(id.matches(adapter.id) && label(adapter.abi) && label(adapter.os) && label(adapter.runtimeRevision) && sha.matches(adapter.artifactSha256))
        }
    }
    private fun sortKey(contract: HyperLFeatureContract) = listOf(contract.featureId, contract.protocol.protocolId,
        contract.protocol.version.toString(), contract.workload.programSha256, contract.workload.shapeSignature,
        contract.workload.dtype, contract.workload.numericalMode).joinToString("\u0000")
}
