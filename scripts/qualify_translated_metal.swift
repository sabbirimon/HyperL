// SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
// Usage: swift scripts/qualify_translated_metal.swift <translated add directory>
import Foundation
import Metal

struct Report: Decodable {
    struct Interface: Decodable { let entry: String }
    let interface: Interface
}
let root = URL(fileURLWithPath: CommandLine.arguments[1])
let report = try JSONDecoder().decode(Report.self, from: Data(contentsOf: root.appendingPathComponent("migration.json")))
let source = try String(contentsOf: root.appendingPathComponent("kernel.metal"), encoding: .utf8)
let devices = MTLCopyAllDevices()
guard !devices.isEmpty else { fatalError("No Metal device") }
for device in devices {
    let lib = try device.makeLibrary(source: source, options: nil)
    guard let entry = lib.makeFunction(name: report.interface.entry) else { fatalError("Entry point missing") }
    let pipeline = try device.makeComputePipelineState(function: entry)
    let n = 137, padded = 192
    let queue = device.makeCommandQueue()!
    let command = queue.makeCommandBuffer()!
    let encoder = command.makeComputeCommandEncoder()!
    let buffers = (0..<3).map { _ in device.makeBuffer(length: padded * 4, options: .storageModeShared)! }
    let ptr = buffers.map { $0.contents().bindMemory(to: Float.self, capacity: padded) }
    for i in 0..<padded {
        ptr[0][i] = Float(i) * 0.5
        ptr[1][i] = Float(100 - i) * 0.25
        ptr[2][i] = -999
    }
    encoder.setComputePipelineState(pipeline)
    for i in 0..<3 { encoder.setBuffer(buffers[i], offset: 0, index: i) }
    var count = UInt32(n)
    encoder.setBytes(&count, length: 4, index: 3)
    let sizes = [UInt32](repeating: UInt32(padded * 4), count: 3)
    sizes.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 30) }
    encoder.dispatchThreadgroups(MTLSize(width: 3, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else { fatalError("Metal command failed: \(String(describing: command.error))") }
    for i in 0..<padded {
        precondition(ptr[2][i] == (i < n ? ptr[0][i] + ptr[1][i] : -999), "Mismatch at \(i)")
    }
    print("\(device.name): \(n) sums and \(padded - n) guard elements passed")
}
