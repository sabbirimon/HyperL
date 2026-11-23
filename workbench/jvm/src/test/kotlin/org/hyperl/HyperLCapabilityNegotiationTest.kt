// SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
// Copyright 2026 Sabbir Hassan Imon and respective contributors.
package org.hyperl

import org.junit.Assert.*
import org.junit.Test

class HyperLCapabilityNegotiationTest {
    private val now = 100_000L
    private val wire = HyperLWireContract("hyperl-shard", 1)
    private val work = HyperLWorkloadContract("a".repeat(64), "vector[16384]", "f32", "ordered-f32-exact/1")
    private val contract = HyperLFeatureContract("elementwise-shards", wire, work)
    private val requested = HyperLRequestedCapability(contract, setOf("multiply", "relu"))
    private fun node(id: String, target: HyperLTarget = HyperLTarget.CPU_REFERENCE,
        hardware: HyperLHardwareClass = HyperLHardwareClass.CPU): HyperLNodeCapabilities {
        val adapter = HyperLAdapterDescriptor("adapter-$id", target, "b".repeat(64), "test-abi", "TestOS", "test-runtime-1",
            AdapterLicenseKind.OPEN_SOURCE, licenseReviewed = true, ownerApproved = true, signatureVerified = true, deviceQualified = true)
        val offer = HyperLFeatureOffer(contract, adapter, hardware, setOf("add", "multiply", "relu"), "test-driver-1", now,
            ownerEnabled = true, qualified = true)
        return HyperLNodeCapabilities(id, "TestOS", "test-abi", now, listOf(offer), ownerApproved = true, identityVerified = true)
    }
    private fun lanes(kind: HyperLFabricKind = HyperLFabricKind.ETHERNET, extension: String? = null) = listOf(
        HyperLFabricLane("lane-1", "left", "right", kind, wire, "transport-left", "driver-1", now, extension, true, true, true),
        HyperLFabricLane("lane-1", "right", "left", kind, wire, "transport-right", "driver-2", now, extension, true, true, true))
    private fun decide(left: HyperLNodeCapabilities = node("left"), right: HyperLNodeCapabilities = node("right"),
        links: List<HyperLFabricLane> = lanes(), requests: List<HyperLRequestedCapability> = listOf(requested)) =
        HyperLCapabilityNegotiation.negotiate(left, right, links, requests, now)
    private fun changeOffer(node: HyperLNodeCapabilities, change: (HyperLFeatureOffer) -> HyperLFeatureOffer) = node.copy(offers = node.offers.map(change))

    @Test fun exactEvidenceEnablesOnlyCommonOperatorsAcrossDifferentProcessorTypes() {
        val right = changeOffer(node("right", HyperLTarget.METAL, HyperLHardwareClass.GPU)) {
            it.copy(supportedOperations = setOf("multiply", "relu"), adapter = it.adapter.copy(abi = "another-abi"))
        }.copy(abi = "another-abi")
        val result = decide(right = right)
        val candidate = result.decisions.single().candidate!!
        assertEquals(setOf("multiply", "relu"), candidate.commonOperations)
        assertEquals(listOf("lane-1"), candidate.eligibleLaneIds)
        assertFalse(result.executionActivated)
        assertFalse(result.resourceReservationGuaranteed)
    }

    @Test fun matchingHardwareNeverBypassesOwnerIdentityFreshnessOrQualifiedAdapterGates() {
        val right = node("right")
        val blocked = listOf(right.copy(ownerApproved = false), right.copy(identityVerified = false),
            right.copy(observedAtMs = now + 1), right.copy(observedAtMs = now - 30_001),
            right.copy(offers = emptyList()), changeOffer(right) { it.copy(ownerEnabled = false) },
            changeOffer(right) { it.copy(qualified = false) }, changeOffer(right) { it.copy(observedAtMs = now + 1) },
            changeOffer(right) { it.copy(observedAtMs = now - 30_001) },
            changeOffer(right) { it.copy(adapter = it.adapter.copy(signatureVerified = false)) },
            changeOffer(right) { it.copy(adapter = it.adapter.copy(deviceQualified = false)) },
            changeOffer(right) { it.copy(adapter = it.adapter.copy(ownerApproved = false)) },
            changeOffer(right) { it.copy(adapter = it.adapter.copy(licenseReviewed = false)) },
            changeOffer(right) { it.copy(adapter = it.adapter.copy(abi = "wrong-abi")) })
        blocked.forEach { assertNull(decide(right = it).decisions.single().candidate) }
        assertNotNull(decide(right = right.copy(observedAtMs = now - 30_000)).decisions.single().candidate)
    }

    @Test fun protocolVersionWorkloadShapeDtypeAndNumericModeMustMatchExactly() {
        val alternatives = listOf(contract.copy(protocol = wire.copy(version = 2)),
            contract.copy(protocol = wire.copy(protocolId = "other-wire")),
            contract.copy(workload = work.copy(shapeSignature = "vector[32768]")),
            contract.copy(workload = work.copy(dtype = "bf16")),
            contract.copy(workload = work.copy(numericalMode = "relaxed-f32/1")),
            contract.copy(workload = work.copy(programSha256 = "c".repeat(64))))
        alternatives.forEach { alternative ->
            assertNull(decide(right = changeOffer(node("right")) { it.copy(contract = alternative) }).decisions.single().candidate)
        }
        assertNull(decide(right = changeOffer(node("right")) { it.copy(supportedOperations = setOf("add")) }).decisions.single().candidate)
        // No mixing an operator from one protocol tuple with a shape/mode from another.
        val one = node("right").offers.single()
        val splitEvidence = node("right").copy(offers = listOf(one.copy(supportedOperations = setOf("multiply")),
            one.copy(contract = contract.copy(protocol = wire.copy(version = 2)), supportedOperations = setOf("relu"))))
        assertNull(decide(right = splitEvidence).decisions.single().candidate)
    }

    @Test fun laneMustHaveFreshApprovedQualifiedAuthenticatedEvidenceInBothDirections() {
        val good = lanes()
        val broken = listOf(emptyList(), good.take(1), listOf(good[0], good[1].copy(authenticated = false)),
            listOf(good[0], good[1].copy(qualified = false)), listOf(good[0].copy(ownerApproved = false), good[1]),
            listOf(good[0], good[1].copy(observedAtMs = now - 30_001)),
            listOf(good[0], good[1].copy(observedAtMs = now + 1)),
            listOf(good[0], good[1].copy(kind = HyperLFabricKind.NVLINK)),
            listOf(good[0], good[1].copy(laneId = "different-lane")),
            listOf(good[0], good[1].copy(protocol = wire.copy(version = 2))))
        broken.forEach { assertNull(decide(links = it).decisions.single().candidate) }
    }

    @Test fun fpgaAndAsicRequireExplicitFirmwareConfigurationQualificationAndExistingBackend() {
        for (hardware in listOf(HyperLHardwareClass.FPGA, HyperLHardwareClass.ASIC)) {
            val target = if (hardware == HyperLHardwareClass.FPGA) HyperLTarget.FPGA_VENDOR else HyperLTarget.NPU_STABLEHLO
            // Fixture declares a separately qualified existing backend; it does not implement ASIC/NPU hardware.
            val right = node("right", target, hardware)
            assertNull(decide(right = right).decisions.single().candidate)
            val firmwareOnly = changeOffer(right) { it.copy(firmwareRevision = "firmware-1") }
            assertNull(decide(right = firmwareOnly).decisions.single().candidate)
            val complete = changeOffer(firmwareOnly) { it.copy(configurationSha256 = "d".repeat(64)) }
            assertNotNull(decide(right = complete).decisions.single().candidate)
            assertNull(decide(right = changeOffer(complete) { it.copy(qualified = false) }).decisions.single().candidate)
            assertNull(decide(right = changeOffer(complete) { it.copy(adapter = it.adapter.copy(target = HyperLTarget.CPU_REFERENCE)) }).decisions.single().candidate)
            val wrongTarget = if (hardware == HyperLHardwareClass.FPGA) HyperLTarget.NPU_STABLEHLO else HyperLTarget.FPGA_VENDOR
            assertNull(decide(right = changeOffer(complete) { it.copy(adapter = it.adapter.copy(target = wrongTarget)) }).decisions.single().candidate)
        }
    }

    @Test fun candidateRetainsExactIndependentOfferAndDirectedLaneSnapshots() {
        val operations = mutableSetOf("multiply", "relu")
        val right = changeOffer(node("right", HyperLTarget.FPGA_VENDOR, HyperLHardwareClass.FPGA)) {
            it.copy(supportedOperations = operations, firmwareRevision = "firmware-2", configurationSha256 = "d".repeat(64))
        }
        val links = lanes().toMutableList()
        val candidate = decide(right = right, links = links).decisions.single().candidate!!
        assertEquals(right.offers.single(), candidate.rightOffer)
        assertEquals("b".repeat(64), candidate.rightOffer.adapter.artifactSha256)
        assertEquals("test-driver-1", candidate.rightOffer.driverRevision)
        assertEquals("firmware-2", candidate.rightOffer.firmwareRevision)
        assertEquals("d".repeat(64), candidate.rightOffer.configurationSha256)
        assertEquals(links.toSet(), candidate.eligibleLanes.toSet())
        assertEquals(setOf("driver-1", "driver-2"), candidate.eligibleLanes.map { it.adapterRevision }.toSet())
        operations.clear()
        links.clear()
        assertEquals(setOf("multiply", "relu"), candidate.rightOffer.supportedOperations)
        assertEquals(2, candidate.eligibleLanes.size)
    }

    @Test fun multipleFabricKindsAndCustomExtensionsAreOnlyClassificationUntilQualified() {
        for (kind in HyperLFabricKind.entries) {
            val extension = if (kind == HyperLFabricKind.CUSTOM) "vendor.fabric/1" else null
            assertFalse(HyperLCapabilityNegotiation.fabricProfile(kind, extension).executionProvided)
            assertNull(decide(links = lanes(kind, extension).map { it.copy(qualified = false) }).decisions.single().candidate)
            assertNotNull(decide(links = lanes(kind, extension)).decisions.single().candidate)
        }
        assertEquals(HyperLFabricScope.ACCELERATOR_SCALE_UP, HyperLCapabilityNegotiation.fabricProfile(HyperLFabricKind.NVLINK).scope)
        assertEquals(HyperLFabricScope.NETWORK, HyperLCapabilityNegotiation.fabricProfile(HyperLFabricKind.INFINIBAND).scope)
        assertEquals(HyperLFabricScope.COHERENT_MEMORY, HyperLCapabilityNegotiation.fabricProfile(HyperLFabricKind.CXL).scope)
        assertThrows(IllegalArgumentException::class.java) { HyperLCapabilityNegotiation.fabricProfile(HyperLFabricKind.CUSTOM) }
        assertThrows(IllegalArgumentException::class.java) { HyperLCapabilityNegotiation.fabricProfile(HyperLFabricKind.USB, "vendor.fabric") }
    }

    @Test fun peerOffersCannotExpandLocalImplementationListAndFutureContractsNeedExactExtension() {
        val future = contract.copy(featureId = "vendor.model-layers", protocol = HyperLWireContract("vendor.model-shards", 7))
        val left = node("left").let { it.copy(offers = it.offers + it.offers.single().copy(contract = future)) }
        val right = node("right").let { it.copy(offers = it.offers + it.offers.single().copy(contract = future)) }
        assertEquals(listOf(contract), decide(left, right).decisions.map { it.contract })
        val futureRequest = HyperLRequestedCapability(future, requested.requiredOperations)
        assertNull(decide(left, right, requests = listOf(futureRequest)).decisions.single().candidate)
        assertNotNull(decide(left, right, lanes(HyperLFabricKind.CUSTOM, "vendor.transport/1").map { it.copy(protocol = future.protocol) },
            listOf(futureRequest)).decisions.single().candidate)
    }

    @Test fun malformedDuplicateAndOversizedInputIsRejectedInsteadOfGrantingPartialResults() {
        assertThrows(IllegalArgumentException::class.java) { decide(left = node("left").copy(deviceId = "right")) }
        assertThrows(IllegalArgumentException::class.java) { decide(links = lanes() + lanes().first()) }
        assertThrows(IllegalArgumentException::class.java) { decide(links = listOf(lanes().first().copy(toDeviceId = "unknown"))) }
        assertThrows(IllegalArgumentException::class.java) { decide(requests = List(17) { requested }) }
        assertThrows(IllegalArgumentException::class.java) { decide(requests = listOf(requested, requested)) }
        assertThrows(IllegalArgumentException::class.java) { decide(right = node("right").copy(offers = List(65) { node("right").offers.single() })) }
        assertThrows(IllegalArgumentException::class.java) { decide(right = changeOffer(node("right")) { it.copy(configurationSha256 = "invalid") }) }
        assertThrows(IllegalArgumentException::class.java) { decide(requests = listOf(requested.copy(requiredOperations = emptySet()))) }
        assertThrows(IllegalArgumentException::class.java) { decide(right = changeOffer(node("right")) { it.copy(contract = contract.copy(protocol = wire.copy(version = 0))) }) }
    }
}
