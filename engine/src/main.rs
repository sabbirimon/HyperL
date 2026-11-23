// SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
use hyperl_engine::{
    archive::{transcode, Codec},
    Projection, Weights,
};
use std::{
    env,
    error::Error,
    io::{self, Write},
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
};
use tempfile::NamedTempFile;

const USAGE: &str = "hyperl-engine processors\nhyperl-engine {plan|project} FILE SHA256 INPUT WEIGHT BYTE_BUDGET {auto|WORKERS|gpu:ADAPTER} [OUTPUT for project] [--int4-scale SCALE]\nhyperl-engine pack {auto|arithmetic|huffman|ans|raw} INPUT OUTPUT MAX_PLAIN_BYTES\nhyperl-engine unpack INPUT OUTPUT MAX_PLAIN_BYTES";

fn publish(path: &Path, shape: [usize; 2], data: &[u8], stop: &AtomicBool) -> io::Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut tmp = NamedTempFile::new_in(parent)?;
    // Fixed schema with integer-only fields. Eight-byte padding follows SafeTensors.
    let mut header = format!(
        r#"{{"output":{{"dtype":"F32","shape":[{},{}],"data_offsets":[0,{}]}}}}"#,
        shape[0],
        shape[1],
        data.len()
    );
    header.extend(std::iter::repeat_n(' ', (8 - header.len() % 8) % 8));
    tmp.write_all(&(header.len() as u64).to_le_bytes())?;
    tmp.write_all(header.as_bytes())?;
    for chunk in data.chunks(1024 * 1024) {
        if stop.load(Ordering::Relaxed) {
            return Err(io::Error::new(io::ErrorKind::Interrupted, "cancelled"));
        }
        tmp.write_all(chunk)?;
    }
    tmp.as_file().sync_all()?;
    if stop.load(Ordering::Relaxed) {
        return Err(io::Error::new(io::ErrorKind::Interrupted, "cancelled"));
    }
    tmp.persist_noclobber(path)?;
    Ok(())
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.as_slice() == ["processors"] {
        println!(
            "backend=cpu logical_processors={} worker_cap=64",
            thread::available_parallelism()?
        );
        #[cfg(feature = "gpu")]
        for name in hyperl_engine::gpu::devices() {
            println!("gpu:{name}");
        }
        return Ok(());
    }
    if args.as_slice() == ["--version"] {
        println!("hyperl-engine {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    if args.as_slice() == ["--help"] {
        println!("{USAGE}");
        return Ok(());
    }
    let stop = Arc::new(AtomicBool::new(false));
    let signal = Arc::clone(&stop);
    ctrlc::set_handler(move || signal.store(true, Ordering::Relaxed))?;
    let archive = match args.as_slice() {
        [cmd, mode, src, dst, max] if cmd == "pack" => {
            let codec = match mode.as_str() {
                "auto" => Codec::Auto,
                "arithmetic" => Codec::Arithmetic,
                "huffman" => Codec::Huffman,
                "ans" => Codec::Ans,
                "raw" => Codec::Raw,
                _ => return Err(USAGE.into()),
            };
            Some((src, dst, max, Some(codec)))
        }
        [cmd, src, dst, max] if cmd == "unpack" => Some((src, dst, max, None)),
        _ => None,
    };
    if let Some((src, dst, max, codec)) = archive {
        transcode(Path::new(src), Path::new(dst), codec, max.parse()?, &stop)?;
        return Ok(());
    }
    let project = match args.first().map(String::as_str) {
        Some("plan") => false,
        Some("project") => true,
        _ => return Err(USAGE.into()),
    };
    let positional = if project { 8 } else { 7 };
    if args.len() != positional && args.len() != positional + 2 {
        return Err(USAGE.into());
    }
    let scale = match &args[positional..] {
        [] => None,
        [flag, value] if flag == "--int4-scale" => Some(value.parse()?),
        _ => return Err(USAGE.into()),
    };
    let mut digest = [0u8; 32];
    let hex = &args[2];
    if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("expected hexadecimal SHA-256".into());
    }
    for (i, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16)?;
    }
    let budget = args[5].parse()?;
    let gpu = args[6].strip_prefix("gpu:");
    let workers = if args[6] == "auto" || gpu.is_some() {
        None
    } else {
        Some(args[6].parse()?)
    };
    // CLI contract: the operator supplies an immutable, read-only artifact.
    let file = unsafe { Weights::open(Path::new(&args[1]), &digest, budget, &stop)? };
    let projection = Projection::prepare(&file, &args[3], &args[4], scale, budget, workers)?;
    if let Some(adapter) = gpu {
        if !project {
            return Err("GPU admission requires project and a device".into());
        }
        #[cfg(feature = "gpu")]
        {
            let data = hyperl_engine::gpu::project(&projection, adapter, budget, &stop)?;
            publish(Path::new(&args[7]), projection.shape(), &data, &stop)?;
            println!("backend={adapter} output_bytes={}", data.len());
            return Ok(());
        }
        #[cfg(not(feature = "gpu"))]
        return Err(format!("GPU feature required for {adapter}").into());
    }
    let plan = projection.plan();
    println!(
        "backend=cpu workers={} tensor_bytes={} output_bytes={}",
        plan.workers, plan.tensor_bytes, plan.output_bytes
    );
    if project {
        let data = projection.execute(&stop)?;
        publish(Path::new(&args[7]), projection.shape(), &data, &stop)?;
    }
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
