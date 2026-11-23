// SPDX-License-Identifier: Apache-2.0
import Foundation
import HyperL

print("=========================================")
print("   HyperL Swift Native Engine CLI        ")
print("=========================================")

print("Engine Version: \(HyperLContext.version)")

do {
    let ctx = try HyperLContext(threads: 4)
    print("Context created successfully with 4 threads")

    let x: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
    let normed = try ctx.rmsnorm(x: x)
    print("RMSNorm result: \(normed)")

    let W = [Float](repeating: 1.0, count: 4 * 8)
    let y = try ctx.gemv(W: W, x: x, M: 4, K: 8)
    print("GEMV result (M=4): \(y)")
    print(">>> Swift Native Engine Execution Verified! <<<")
} catch {
    print("Error: \(error)")
}
