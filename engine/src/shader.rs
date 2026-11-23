// SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
use naga::{
    back::{hlsl, msl, spv, wgsl},
    valid::{Capabilities, ValidationFlags, Validator},
};
use std::{
    error::Error,
    fs::File,
    io::{Read, Write},
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};
use tempfile::NamedTempFile;

pub fn translate(
    input: &Path,
    target: &str,
    output: &Path,
    stop: &AtomicBool,
) -> Result<(), Box<dyn Error>> {
    let mut source = String::new();
    let file = File::open(input)?;
    if !file.metadata()?.is_file() {
        return Err("choose a regular WGSL source file".into());
    }
    file.take(1_048_577).read_to_string(&mut source)?;
    if source.len() > 1_048_576 {
        return Err("shader exceeds 1 MiB".into());
    }
    let module = naga::front::wgsl::parse_str(&source).map_err(|e| e.emit_to_string(&source))?;
    if module.entry_points.is_empty()
        || module
            .entry_points
            .iter()
            .any(|ep| ep.stage != naga::ShaderStage::Compute)
    {
        return Err("expected a compute shader".into());
    }
    let info = Validator::new(ValidationFlags::all(), Capabilities::empty()).validate(&module)?;
    let data = match target {
        "wgsl" => wgsl::write_string(&module, &info, wgsl::WriterFlags::empty())?.into_bytes(),
        "spirv" => spv::write_vec(&module, &info, &Default::default(), None)?
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect(),
        "hlsl" => {
            let mut text = String::new();
            let options = hlsl::Options::default();
            let reflection = hlsl::Writer::new(&mut text, &options).write(&module, &info, None)?;
            for (original, actual) in module.entry_points.iter().zip(reflection.entry_point_names) {
                println!("entry {} {}", original.name, actual?);
            }
            text.into_bytes()
        }
        "metal" => {
            let mut resources = msl::EntryPointResources {
                sizes_buffer: Some(30),
                ..Default::default()
            };
            for (_, var) in module.global_variables.iter() {
                if let Some(binding) = &var.binding {
                    if binding.group != 0
                        || binding.binding >= 30
                        || !matches!(
                            var.space,
                            naga::AddressSpace::Uniform | naga::AddressSpace::Storage { .. }
                        )
                    {
                        return Err(
                            "Metal translation accepts group 0 buffer bindings 0 through 29".into(),
                        );
                    }
                    resources.resources.insert(binding.clone(), msl::BindTarget {
                        buffer: Some(binding.binding as u8),
                        mutable: matches!(var.space, naga::AddressSpace::Storage { access } if access.contains(naga::StorageAccess::STORE)),
                        ..Default::default()
                    });
                }
            }
            let options = msl::Options {
                lang_version: (2, 1),
                fake_missing_bindings: false,
                per_entry_point_map: module
                    .entry_points
                    .iter()
                    .map(|ep| (ep.name.clone(), resources.clone()))
                    .collect(),
                ..Default::default()
            };
            let (text, reflection) =
                msl::write_string(&module, &info, &options, &Default::default())?;
            for (original, actual) in module.entry_points.iter().zip(reflection.entry_point_names) {
                println!("entry {} {}", original.name, actual?);
            }
            text.into_bytes()
        }
        _ => return Err("choose wgsl, spirv, metal or hlsl".into()),
    };
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut tmp = NamedTempFile::new_in(parent)?;
    tmp.write_all(&data)?;
    tmp.as_file().sync_all()?;
    if stop.load(Ordering::Relaxed) {
        return Err("cancelled".into());
    }
    tmp.persist_noclobber(output)?;
    Ok(())
}
