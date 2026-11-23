// SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
//! HLZ1 uses independent 64 KiB blocks, an integer histogram, and two
//! end-to-end digests over the encoded framing and original artifact.
use super::{checkpoint, invalid};
use constriction::{
    backends::Cursor,
    stream::{
        model::DefaultContiguousCategoricalEntropyModel as Model,
        queue::{DefaultRangeDecoder, DefaultRangeEncoder},
        stack::DefaultAnsCoder,
        Decode, Encode,
    },
    symbol::{
        huffman::{DecoderHuffmanTree, EncoderHuffmanTree},
        DefaultQueueDecoder, DefaultQueueEncoder, ReadBitStream, WriteBitStream,
    },
};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{self, Read, Write},
    path::Path,
    sync::atomic::AtomicBool,
};
use tempfile::NamedTempFile;

const BLOCK: usize = 65536;
const MAGIC: &[u8; 4] = b"HLZ1";
#[derive(Clone, Copy, PartialEq)]
pub enum Codec {
    Auto,
    Raw,
    Arithmetic,
    Huffman,
    Ans,
}

fn model(counts: &[u32; 256], len: usize) -> io::Result<Model> {
    // Every byte gets at least one of 2^24 slots. Integer cumulative
    // rounding gives the exact same table on every supported architecture.
    let mut prefix = 0u64;
    let mut previous = 0u64;
    let frequencies = counts.iter().map(|&count| {
        prefix += u64::from(count);
        let end = prefix * ((1 << 24) - 256) / (len as u64);
        let frequency = end - previous + 1;
        previous = end;
        frequency as u32
    });
    Model::from_nonzero_fixed_point_probabilities(frequencies, false)
        .map_err(|_| invalid("invalid coding model"))
}

fn encode(codec: Codec, input: &[u8], counts: &[u32; 256]) -> io::Result<(u8, Vec<u8>)> {
    let symbols = input.iter().map(|&x| usize::from(x));
    let (tag, words) = match codec {
        Codec::Huffman => {
            let tree = EncoderHuffmanTree::from_probabilities::<u32, _>(counts.map(|n| n + 1));
            let mut coder = DefaultQueueEncoder::new();
            coder
                .encode_iid_symbols(symbols, &tree)
                .map_err(|_| invalid("Huffman encoding failed"))?;
            (1, coder.into_compressed().unwrap())
        }
        Codec::Arithmetic => {
            let mut coder = DefaultRangeEncoder::new();
            coder
                .encode_iid_symbols(symbols, &model(counts, input.len())?)
                .map_err(|_| invalid("range encoding failed"))?;
            (2, coder.into_compressed().unwrap())
        }
        Codec::Ans => {
            let mut coder = DefaultAnsCoder::new();
            coder
                .encode_iid_symbols_reverse(symbols, &model(counts, input.len())?)
                .map_err(|_| invalid("ANS encoding failed"))?;
            (3, coder.into_compressed().unwrap())
        }
        _ => return Ok((0, input.to_vec())),
    };
    let bytes = counts
        .iter()
        .chain(&words)
        .flat_map(|v| v.to_le_bytes())
        .collect();
    Ok((tag, bytes))
}

fn decode(tag: u8, payload: Vec<u8>, len: usize) -> io::Result<Vec<u8>> {
    if tag == 0 {
        return if payload.len() == len {
            Ok(payload)
        } else {
            Err(invalid("wrong stored length"))
        };
    }
    if !(1..=3).contains(&tag) || payload.len() < 1024 || payload.len() % 4 != 0 {
        return Err(invalid("invalid coded block"));
    }
    let mut words = payload
        .chunks_exact(4)
        .map(|x| u32::from_le_bytes(x.try_into().unwrap()));
    let counts: [u32; 256] = std::array::from_fn(|_| words.next().unwrap());
    if counts.iter().map(|&n| u64::from(n)).sum::<u64>() != len as u64 {
        return Err(invalid("invalid histogram total"));
    }
    let data: Vec<_> = words.collect();
    let mut result = Vec::with_capacity(len);
    match tag {
        1 => {
            let tree = DecoderHuffmanTree::from_probabilities::<u32, _>(counts.map(|n| n + 1));
            let mut coder =
                DefaultQueueDecoder::from_compressed(Cursor::new_at_write_beginning(data));
            for sym in coder.decode_iid_symbols(len, &tree) {
                result.push(sym.map_err(|_| invalid("invalid Huffman stream"))? as u8);
            }
        }
        2 => {
            let mut coder = DefaultRangeDecoder::from_compressed(data).unwrap();
            for sym in coder.decode_iid_symbols(len, &model(&counts, len)?) {
                result.push(sym.map_err(|_| invalid("invalid range stream"))? as u8);
            }
        }
        3 => {
            let mut coder =
                DefaultAnsCoder::from_compressed(data).map_err(|_| invalid("invalid ANS state"))?;
            for sym in coder.decode_iid_symbols(len, &model(&counts, len)?) {
                result.push(sym.map_err(|_| invalid("invalid ANS stream"))? as u8);
            }
            if !coder.is_empty() {
                return Err(invalid("unconsumed ANS state"));
            }
        }
        _ => unreachable!(),
    }
    Ok(result)
}

fn read_array<const N: usize>(f: &mut File) -> io::Result<[u8; N]> {
    let mut a = [0; N];
    f.read_exact(&mut a)?;
    Ok(a)
}

/// A successful return publishes a complete new destination. The limit applies
/// to decoded bytes. Block framing separately bounds all decoder allocations.
pub fn transcode(
    input: &Path,
    output: &Path,
    pack: Option<Codec>,
    limit: u64,
    stop: &AtomicBool,
) -> io::Result<()> {
    let mut f = File::open(input)?;
    let stat = f.metadata()?;
    if !stat.is_file() {
        return Err(invalid("regular input required"));
    }
    let size = if pack.is_some() {
        stat.len()
    } else {
        if &read_array::<4>(&mut f)? != MAGIC {
            return Err(invalid("unknown archive version"));
        }
        u64::from_le_bytes(read_array(&mut f)?)
    };
    if size > limit {
        return Err(invalid("decoded size exceeds budget"));
    }
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut out = NamedTempFile::new_in(parent)?;
    let mut plain = Sha256::new();
    let mut wire = Sha256::new();
    wire.update(MAGIC);
    wire.update(size.to_le_bytes());
    if pack.is_some() {
        out.write_all(MAGIC)?;
        out.write_all(&size.to_le_bytes())?;
    }
    let mut remaining = size;
    while remaining > 0 {
        checkpoint(stop)?;
        let len = remaining.min(BLOCK as u64) as usize;
        if let Some(codec) = pack {
            let mut data = vec![0; len];
            f.read_exact(&mut data)?;
            plain.update(&data);
            let mut counts = [0u32; 256];
            for &byte in &data {
                counts[byte as usize] += 1;
            }
            let (tag, payload) = if codec == Codec::Auto {
                let mut best = (0, data.clone());
                for candidate in [Codec::Huffman, Codec::Arithmetic, Codec::Ans] {
                    let encoded = encode(candidate, &data, &counts)?;
                    if encoded.1.len() < best.1.len() {
                        best = encoded;
                    }
                }
                best
            } else {
                encode(codec, &data, &counts)?
            };
            let bytes = (payload.len() as u32).to_le_bytes();
            out.write_all(&[tag])?;
            out.write_all(&bytes)?;
            out.write_all(&payload)?;
            wire.update([tag]);
            wire.update(bytes);
            wire.update(payload);
        } else {
            let [tag] = read_array(&mut f)?;
            let bytes = read_array(&mut f)?;
            let n = u32::from_le_bytes(bytes) as usize;
            if n > 1024 + 4 * len + 8 {
                return Err(invalid("coded payload exceeds block limit"));
            }
            let mut payload = vec![0; n];
            f.read_exact(&mut payload)?;
            wire.update([tag]);
            wire.update(bytes);
            wire.update(&payload);
            let decoded = decode(tag, payload, len)?;
            plain.update(&decoded);
            out.write_all(&decoded)?;
        }
        remaining -= len as u64;
    }
    for hash in [wire.finalize(), plain.finalize()] {
        if pack.is_some() {
            out.write_all(&hash)?;
        } else if hash[..] != read_array::<32>(&mut f)? {
            return Err(invalid("archive integrity failure"));
        }
    }
    if f.read(&mut [0])? != 0 {
        return Err(invalid("unexpected trailing data"));
    }
    out.as_file().sync_all()?;
    checkpoint(stop)?;
    out.persist_noclobber(output)?;
    Ok(())
}
