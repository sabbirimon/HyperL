// SPDX-License-Identifier: Apache-2.0
// HyperL Native Inference Engine Swift API

import Foundation
import CHyperL

public enum HyperLError: Error {
    case invalidArgument
    case outOfMemory
    case executionFailed(Int32)
}

public final class HyperLContext {
    private var raw: OpaquePointer?

    public init(device: hl_device_type_t = HL_DEVICE_CPU, threads: Int32 = 4) throws {
        var ctx: OpaquePointer?
        let status = hl_context_create(&ctx, device, threads)
        guard status == HL_SUCCESS, let validCtx = ctx else {
            throw HyperLError.executionFailed(Int32(status.rawValue))
        }
        self.raw = validCtx
    }

    deinit {
        if let ctx = raw {
            hl_context_destroy(ctx)
            raw = nil
        }
    }

    public static var version: String {
        return String(cString: hl_engine_version())
    }

    public func rmsnorm(x: [Float], weight: [Float]? = nil, eps: Float = 1e-5) throws -> [Float] {
        guard !x.isEmpty else { throw HyperLError.invalidArgument }
        var out = [Float](repeating: 0, count: x.count)

        var desc = hl_tensor_desc_t()
        desc.dtype = HL_DTYPE_F32
        desc.ndim = 1
        desc.shape.0 = Int64(x.count)

        try x.withUnsafeBufferPointer { xPtr in
            try out.withUnsafeMutableBufferPointer { outPtr in
                var tx = hl_tensor_t(desc: desc, data: UnsafeMutableRawPointer(mutating: xPtr.baseAddress), device: HL_DEVICE_CPU, is_view: true, device_buffer_handle: nil)
                var tout = hl_tensor_t(desc: desc, data: outPtr.baseAddress, device: HL_DEVICE_CPU, is_view: true, device_buffer_handle: nil)

                let status: hl_status_t
                if let w = weight, w.count == x.count {
                    var tw = w.withUnsafeBufferPointer { wPtr in
                        hl_tensor_t(desc: desc, data: UnsafeMutableRawPointer(mutating: wPtr.baseAddress), device: HL_DEVICE_CPU, is_view: true, device_buffer_handle: nil)
                    }
                    status = hl_kernel_rmsnorm(raw, &tx, &tw, eps, &tout)
                } else {
                    status = hl_kernel_rmsnorm(raw, &tx, nil, eps, &tout)
                }

                guard status == HL_SUCCESS else {
                    throw HyperLError.executionFailed(Int32(status.rawValue))
                }
            }
        }
        return out
    }

    public func gemv(W: [Float], x: [Float], M: Int, K: Int) throws -> [Float] {
        guard W.count == M * K && x.count == K else { throw HyperLError.invalidArgument }
        var y = [Float](repeating: 0, count: M)

        var descW = hl_tensor_desc_t()
        descW.dtype = HL_DTYPE_F32
        descW.ndim = 2
        descW.shape.0 = Int64(M)
        descW.shape.1 = Int64(K)

        var descX = hl_tensor_desc_t()
        descX.dtype = HL_DTYPE_F32
        descX.ndim = 1
        descX.shape.0 = Int64(K)

        var descY = hl_tensor_desc_t()
        descY.dtype = HL_DTYPE_F32
        descY.ndim = 1
        descY.shape.0 = Int64(M)

        try W.withUnsafeBufferPointer { wPtr in
            try x.withUnsafeBufferPointer { xPtr in
                try y.withUnsafeMutableBufferPointer { yPtr in
                    var tW = hl_tensor_t(desc: descW, data: UnsafeMutableRawPointer(mutating: wPtr.baseAddress), device: HL_DEVICE_CPU, is_view: true, device_buffer_handle: nil)
                    var tx = hl_tensor_t(desc: descX, data: UnsafeMutableRawPointer(mutating: xPtr.baseAddress), device: HL_DEVICE_CPU, is_view: true, device_buffer_handle: nil)
                    var ty = hl_tensor_t(desc: descY, data: yPtr.baseAddress, device: HL_DEVICE_CPU, is_view: true, device_buffer_handle: nil)

                    let status = hl_kernel_gemv(raw, &tW, &tx, nil, &ty)
                    guard status == HL_SUCCESS else {
                        throw HyperLError.executionFailed(Int32(status.rawValue))
                    }
                }
            }
        }
        return y
    }
}
