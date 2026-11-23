// SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
//! Native, bounded CPU projections over immutable mapped model artifacts.
use memmap2::{Mmap, MmapOptions};
use safetensors::tensor::{Dtype, Metadata, SafeTensors, TensorInfo};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    thread,
};

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn checkpoint(stop: &AtomicBool) -> io::Result<()> {
    if stop.load(Ordering::Relaxed) {
        Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "execution cancelled",
        ))
    } else {
        Ok(())
    }
}

pub struct Weights {
    file: Mmap,
    metadata: Metadata,
    base: usize,
}

impl Weights {
    /// Map an owner-controlled, immutable, local regular file. Hash the mapped
    /// bytes before interpreting the header or exposing any tensor.
    ///
    /// # Safety
    /// All writers must be excluded from changing or truncating the underlying
    /// file until this object and its borrowed views have been dropped.
    pub unsafe fn open(
        path: &Path,
        digest: &[u8; 32],
        limit: usize,
        stop: &AtomicBool,
    ) -> io::Result<Self> {
        let file = File::open(path)?;
        let stat = file.metadata()?;
        let size =
            usize::try_from(stat.len()).map_err(|_| invalid("file exceeds address space"))?;
        if !stat.is_file() || !(10..=limit.min(isize::MAX as usize)).contains(&size) {
            return Err(invalid("expected regular file within mapping budget"));
        }
        let map = unsafe { MmapOptions::new().len(size).map(&file)? };
        let mut sha = Sha256::new();
        for chunk in map.chunks(1024 * 1024) {
            checkpoint(stop)?;
            sha.update(chunk);
        }
        if &sha.finalize()[..] != digest {
            return Err(invalid("artifact SHA-256 mismatch"));
        }
        let header_len = u64::from_le_bytes(map[..8].try_into().unwrap());
        if header_len > 4 * 1024 * 1024 {
            return Err(invalid("header exceeds 4 MiB"));
        }
        let (header_len, metadata) = SafeTensors::read_metadata(&map)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        Ok(Self {
            file: map,
            metadata,
            base: 8 + header_len,
        })
    }

    pub fn mapped_bytes(&self) -> usize {
        self.file.len()
    }

    /// Byte slices borrow the original mapping, including unaligned tensors.
    pub fn tensor(&self, name: &str) -> io::Result<(&TensorInfo, &[u8])> {
        let info = self
            .metadata
            .info(name)
            .ok_or_else(|| invalid("unknown tensor name"))?;
        let (start, end) = info.data_offsets;
        Ok((info, &self.file[self.base + start..self.base + end]))
    }
}

/// Workers own disjoint output cells, including batch-one decode. Output elements are always f32.
/// mapped_bytes includes every tensor in the backing file exactly once.
#[derive(Debug, PartialEq)]
pub struct Plan {
    pub workers: usize,
    pub output_bytes: usize,
    pub tensor_bytes: usize,
}

impl Plan {
    pub fn for_projection(
        rows: usize,
        outputs: usize,
        mapped: usize,
        budget: usize,
        available: usize,
        requested: Option<usize>,
    ) -> io::Result<Self> {
        if rows == 0 || outputs == 0 || available == 0 {
            return Err(invalid(
                "positive dimensions and processor capacity required",
            ));
        }
        let cells = rows
            .checked_mul(outputs)
            .ok_or_else(|| invalid("output size overflow"))?;
        let capacity = available.min(64);
        let workers = match requested {
            Some(0) => return Err(invalid("worker count must be positive")),
            Some(n) if n > capacity => {
                return Err(invalid("requested workers exceed local capacity"))
            }
            Some(n) => n.min(cells),
            None => capacity.min(cells),
        };
        let output_bytes = rows
            .checked_mul(outputs)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| invalid("output size overflow"))?;
        let tensor_bytes = mapped
            .checked_add(output_bytes)
            .ok_or_else(|| invalid("tensor budget overflow"))?;
        if tensor_bytes > budget || output_bytes > isize::MAX as usize {
            return Err(invalid("projection exceeds tensor byte budget"));
        }
        Ok(Self {
            workers,
            output_bytes,
            tensor_bytes,
        })
    }
}

#[derive(Clone, Copy)]
enum Encoding {
    F32,
    Bf16,
    E4m3,
    E5m2,
    Int4(f32),
}

struct Matrix<'a> {
    encoding: Encoding,
    data: &'a [u8],
    rows: usize,
    cols: usize,
}

impl<'a> Matrix<'a> {
    fn load(file: &'a Weights, name: &str, int4_scale: Option<f32>) -> io::Result<Self> {
        let (info, data) = file.tensor(name)?;
        let &[rows, mut cols] = info.shape.as_slice() else {
            return Err(invalid("expected a matrix"));
        };
        if rows == 0 || cols == 0 {
            return Err(invalid("empty matrix"));
        }
        let encoding = match (info.dtype, int4_scale) {
            (Dtype::F32, None) => Encoding::F32,
            (Dtype::BF16, None) => Encoding::Bf16,
            (Dtype::F8_E4M3, None) => Encoding::E4m3,
            (Dtype::F8_E5M2, None) => Encoding::E5m2,
            (Dtype::U8, Some(s)) if s.is_finite() && s > 0.0 => {
                cols = cols
                    .checked_mul(2)
                    .ok_or_else(|| invalid("packed width overflow"))?;
                Encoding::Int4(s)
            }
            _ => return Err(invalid("unsupported dtype or packed INT4 scale")),
        };
        rows.checked_mul(cols)
            .ok_or_else(|| invalid("matrix dimensions overflow"))?;
        Ok(Self {
            encoding,
            data,
            rows,
            cols,
        })
    }

    fn value(&self, index: usize) -> f32 {
        match self.encoding {
            Encoding::F32 => {
                f32::from_le_bytes(self.data[index * 4..index * 4 + 4].try_into().unwrap())
            }
            Encoding::Bf16 => {
                let bits =
                    u16::from_le_bytes(self.data[index * 2..index * 2 + 2].try_into().unwrap());
                f32::from_bits(u32::from(bits) << 16)
            }
            Encoding::E4m3 => fp8(self.data[index], 3, 7, true),
            Encoding::E5m2 => fp8(self.data[index], 2, 15, false),
            Encoding::Int4(scale) => {
                let nibble = (self.data[index / 2] >> ((index % 2) * 4)) & 15;
                f32::from(((nibble << 4) as i8) >> 4) * scale
            }
        }
    }
}

fn fp8(bits: u8, fraction_bits: u8, bias: i32, finite_format: bool) -> f32 {
    let exponent = (bits & 127) >> fraction_bits;
    let fraction = bits & ((1 << fraction_bits) - 1);
    let sign = if bits & 128 == 0 { 1.0 } else { -1.0 };
    if !finite_format && exponent == 31 {
        return if fraction == 0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        };
    }
    if finite_format && bits & 127 == 127 {
        return f32::NAN;
    }
    let significand = f32::from(fraction) / ((1u32 << fraction_bits) as f32);
    if exponent == 0 {
        sign * significand * 2f32.powi(1 - bias)
    } else {
        sign * (1.0 + significand) * 2f32.powi(i32::from(exponent) - bias)
    }
}

pub struct Projection<'a> {
    x: Matrix<'a>,
    w: Matrix<'a>,
    plan: Plan,
}

impl<'a> Projection<'a> {
    pub fn prepare(
        file: &'a Weights,
        input: &str,
        weights: &str,
        scale: Option<f32>,
        budget: usize,
        workers: Option<usize>,
    ) -> io::Result<Self> {
        let x = Matrix::load(file, input, None)?;
        let w = Matrix::load(file, weights, scale)?;
        if x.cols != w.cols {
            return Err(invalid("projection widths differ"));
        }
        let available = thread::available_parallelism()?.get();
        let plan = Plan::for_projection(
            x.rows,
            w.rows,
            file.mapped_bytes(),
            budget,
            available,
            workers,
        )?;
        Ok(Self { x, w, plan })
    }
    pub fn plan(&self) -> &Plan {
        &self.plan
    }
    pub fn shape(&self) -> [usize; 2] {
        [self.x.rows, self.w.rows]
    }

    /// Returns little-endian F32 bytes ready for SafeTensors output. No
    /// intermediate dequantized matrix, output conversion or input clone.
    pub fn execute(&self, stop: &AtomicBool) -> io::Result<Vec<u8>> {
        checkpoint(stop)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(self.plan.output_bytes)
            .map_err(io::Error::other)?;
        result.resize(self.plan.output_bytes, 0);
        let chunk_cells = (result.len() / 4).div_ceil(self.plan.workers);
        thread::scope(|scope| {
            let mut jobs = Vec::new();
            for (group, chunk) in result.chunks_mut(chunk_cells * 4).enumerate() {
                jobs.push(thread::Builder::new().spawn_scoped(
                    scope,
                    move || -> io::Result<()> {
                        for (i, cell) in chunk.chunks_exact_mut(4).enumerate() {
                            let linear = group * chunk_cells + i;
                            let offset = (linear / self.w.rows) * self.x.cols;
                            let c = linear % self.w.rows;
                            let mut sum = 0f32;
                            for k in 0..self.x.cols {
                                if k % 1024 == 0 {
                                    checkpoint(stop)?;
                                }
                                sum += self.x.value(offset + k) * self.w.value(c * self.w.cols + k);
                            }
                            if !sum.is_finite() {
                                return Err(invalid("nonfinite projection"));
                            }
                            cell.copy_from_slice(&sum.to_le_bytes());
                        }
                        Ok(())
                    },
                )?);
            }
            for job in jobs {
                job.join()
                    .map_err(|_| io::Error::other("worker panicked"))??;
            }
            checkpoint(stop)
        })?;
        Ok(result)
    }
}

pub mod archive;

#[cfg(feature = "gpu")]
pub mod gpu;
