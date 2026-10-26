package org.hyperl

import java.math.BigDecimal
import java.util.Random
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test

/** Independent exact-decimal oracle for bounded dyadic fixtures; not a universal error proof. */
class PreciseReductionOracleTest {
    @Test fun compensatedReductionMatchesExactDyadicOracleOnCancellationFixtures() = runBlocking {
        val random = Random(721)
        repeat(200) {
            val values = ArrayList<Float>()
            repeat(16) {
                val magnitude = Math.scalb(1f, 24 + random.nextInt(19))
                values.add(magnitude); values.add(-magnitude)
                values.add((random.nextInt(2001) - 1000).toFloat())
            }
            java.util.Collections.shuffle(values, random)
            // Constructing BigDecimal from Double preserves the exact binary32 value.
            val exact = values.fold(BigDecimal.ZERO) { sum, x -> sum.add(BigDecimal(x.toDouble())) }
            assertEquals(exact.toFloat(), PreciseReduction.sum(values.toFloatArray()), 0f)
        }
    }
    @Test fun subnormalResidualAndLargeCancellationRemainSeparateFromGraphSemantics() = runBlocking {
        val cases = listOf(
            floatArrayOf(1f, Float.MIN_VALUE, -1f),
            floatArrayOf(Float.MAX_VALUE, 1f, -Float.MAX_VALUE),
            floatArrayOf(-Float.MAX_VALUE, -1f, Float.MAX_VALUE)
        )
        for(values in cases) {
            val exact = values.fold(BigDecimal.ZERO) { sum, x -> sum.add(BigDecimal(x.toDouble())) }
            assertEquals(exact.toFloat(), PreciseReduction.sum(values), 0f)
        }
    }
}
