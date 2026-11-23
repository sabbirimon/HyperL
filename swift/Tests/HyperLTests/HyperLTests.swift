// SPDX-License-Identifier: Apache-2.0

import XCTest
@testable import HyperL

final class HyperLTests: XCTestCase {
    func testVersionAndContext() throws {
        let v = HyperLContext.version
        XCTAssertFalse(v.isEmpty)
        print("Swift HyperL Version: \(v)")

        let ctx = try HyperLContext(threads: 2)
        XCTAssertNotNil(ctx)
    }

    func testRMSNorm() throws {
        let ctx = try HyperLContext(threads: 2)
        let x: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let out = try ctx.rmsnorm(x: x)
        XCTAssertEqual(out.count, 8)
        XCTAssertEqual(out[0], 0.1980295, accuracy: 1e-4)
    }

    func testGEMV() throws {
        let ctx = try HyperLContext(threads: 2)
        let W = [Float](repeating: 1.0, count: 4 * 8)
        let x: [Float] = [1, 2, 3, 4, 5, 6, 7, 8] // sum = 36
        let y = try ctx.gemv(W: W, x: x, M: 4, K: 8)
        XCTAssertEqual(y.count, 4)
        for val in y {
            XCTAssertEqual(val, 36.0, accuracy: 1e-4)
        }
    }
}
