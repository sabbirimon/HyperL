// SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
struct Shape { rows: u32, cols: u32, width: u32, reserved: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> shape: Shape;
var<workgroup> left: array<array<f32, 8>, 8>;
var<workgroup> right: array<array<f32, 8>, 8>;

@compute @workgroup_size(8, 8)
fn project(@builtin(local_invocation_id) lane: vec3<u32>,
           @builtin(workgroup_id) group: vec3<u32>) {
    let row = group.y * 8u + lane.y;
    let col = group.x * 8u + lane.x;
    let weight_row = group.x * 8u + lane.y;
    var sum = 0.0;
    for (var base = 0u; base < shape.width; base += 8u) {
        let k = base + lane.x;
        var a = 0.0;
        var b = 0.0;
        if row < shape.rows && k < shape.width { a = x[row * shape.width + k]; }
        if weight_row < shape.cols && k < shape.width { b = weights[weight_row * shape.width + k]; }
        left[lane.y][lane.x] = a;
        right[lane.y][lane.x] = b;
        workgroupBarrier();
        for (var j = 0u; j < 8u; j++) { sum += left[lane.y][j] * right[lane.x][j]; }
        workgroupBarrier();
    }
    if row < shape.rows && col < shape.cols { output[row * shape.cols + col] = sum; }
}
