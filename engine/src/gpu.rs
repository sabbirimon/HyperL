// SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
use super::{checkpoint, invalid, Encoding, Projection};
use std::{
    io,
    sync::{atomic::AtomicBool, mpsc},
};
use wgpu::{self as w, util::DeviceExt};

fn adapters() -> Vec<(String, w::Adapter)> {
    let backends = w::Backends::METAL | w::Backends::VULKAN | w::Backends::DX12;
    let instance = w::Instance::new(&w::InstanceDescriptor {
        backends,
        ..Default::default()
    });
    instance
        .enumerate_adapters(backends)
        .into_iter()
        .filter_map(|adapter| {
            let info = adapter.get_info();
            if !matches!(
                info.device_type,
                w::DeviceType::IntegratedGpu | w::DeviceType::DiscreteGpu
            ) {
                return None;
            }
            Some((format!("{:?}/{}", info.backend, info.name), adapter))
        })
        .collect()
}

pub fn devices() -> Vec<String> {
    adapters().into_iter().map(|(name, _)| name).collect()
}

/// Execute on a single, explicitly named physical adapter. The byte budget
/// includes the mapping, result, GPU inputs/output, readback and parameters.
pub fn project(
    p: &Projection<'_>,
    selected: &str,
    budget: usize,
    stop: &AtomicBool,
) -> io::Result<Vec<u8>> {
    if !matches!((p.x.encoding, p.w.encoding), (Encoding::F32, Encoding::F32)) {
        return Err(invalid("GPU projection requires F32 inputs"));
    }
    let size = p.plan.output_bytes;
    let charged = [
        p.plan.tensor_bytes,
        p.x.data.len(),
        p.w.data.len(),
        size,
        size,
        16,
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .ok_or_else(|| invalid("GPU buffer size overflow"))?;
    if charged > budget {
        return Err(invalid("GPU tensor and transfer buffers exceed budget"));
    }
    let mut matches = adapters().into_iter().filter(|(name, _)| name == selected);
    let (_, adapter) = matches
        .next()
        .ok_or_else(|| invalid("physical adapter unavailable"))?;
    if matches.next().is_some() {
        return Err(invalid("ambiguous physical adapter name"));
    }
    let (device, queue) =
        pollster::block_on(adapter.request_device(&w::DeviceDescriptor::default(), None))
            .map_err(io::Error::other)?;
    let limits = device.limits();
    let limit = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
    if [p.x.data.len(), p.w.data.len(), size]
        .iter()
        .any(|&n| n as u64 > limit)
    {
        return Err(invalid("storage binding exceeds device limit"));
    }
    let [rows, cols, width] = [p.x.rows, p.w.rows, p.x.cols].map(|n| n as u32);
    let groups = [cols.div_ceil(8), rows.div_ceil(8)];
    if groups
        .iter()
        .any(|&n| n > limits.max_compute_workgroups_per_dimension)
        || u64::from(rows) * u64::from(cols) * u64::from(width) > 1 << 30
    {
        return Err(invalid(
            "dispatch exceeds device or single-dispatch work bound",
        ));
    }
    for input in [p.x.data, p.w.data] {
        for chunk in input.chunks(64 * 1024) {
            checkpoint(stop)?;
            if chunk
                .chunks_exact(4)
                .any(|b| !f32::from_le_bytes(b.try_into().unwrap()).is_finite())
            {
                return Err(invalid("nonfinite GPU input"));
            }
        }
    }
    device.push_error_scope(w::ErrorFilter::OutOfMemory);
    device.push_error_scope(w::ErrorFilter::Validation);
    let upload = |label, contents, usage| {
        device.create_buffer_init(&w::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage,
        })
    };
    let x = upload("input", p.x.data, w::BufferUsages::STORAGE);
    let weights = upload("weight", p.w.data, w::BufferUsages::STORAGE);
    let shape: Vec<u8> = [rows, cols, width, 0]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    let params = upload("shape", &shape, w::BufferUsages::UNIFORM);
    let buffer = |usage| {
        device.create_buffer(&w::BufferDescriptor {
            label: None,
            size: size as u64,
            usage,
            mapped_at_creation: false,
        })
    };
    let result = buffer(w::BufferUsages::STORAGE | w::BufferUsages::COPY_SRC);
    let readback = buffer(w::BufferUsages::COPY_DST | w::BufferUsages::MAP_READ);
    let shader = device.create_shader_module(w::ShaderModuleDescriptor {
        label: Some("HyperL projection"),
        source: w::ShaderSource::Wgsl(include_str!("projection.wgsl").into()),
    });
    let pipeline = device.create_compute_pipeline(&w::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("project"),
        compilation_options: Default::default(),
        cache: None,
    });
    if let Some(err) = pollster::block_on(device.pop_error_scope()) {
        return Err(io::Error::other(err));
    }
    device.push_error_scope(w::ErrorFilter::Validation);
    let bindings = [&x, &weights, &result, &params];
    let entries: Vec<_> = bindings
        .iter()
        .enumerate()
        .map(|(i, b)| w::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    let bind = device.create_bind_group(&w::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups[0], groups[1], 1);
    }
    encoder.copy_buffer_to_buffer(&result, 0, &readback, 0, size as u64);
    checkpoint(stop)?;
    queue.submit(Some(encoder.finish()));
    let slice = readback.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(w::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(w::Maintain::Wait);
    for _ in 0..2 {
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            return Err(io::Error::other(err));
        }
    }
    rx.recv()
        .map_err(io::Error::other)?
        .map_err(io::Error::other)?;
    checkpoint(stop)?;
    let data = slice.get_mapped_range();
    if data
        .chunks_exact(4)
        .any(|b| !f32::from_le_bytes(b.try_into().unwrap()).is_finite())
    {
        return Err(invalid("nonfinite GPU result"));
    }
    let bytes = data.to_vec();
    drop(data);
    readback.unmap();
    Ok(bytes)
}
