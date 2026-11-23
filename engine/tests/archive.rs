// SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
use hyperl_engine::archive::{transcode, Codec};
use std::{fs, sync::atomic::AtomicBool};
use tempfile::tempdir;

#[test]
fn round_trip_empty_single_symbol_multiblock_and_uniform_bytes() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input");
    let archive = dir.path().join("archive");
    let output = dir.path().join("output");
    let stop = AtomicBool::new(false);
    for data in [
        Vec::new(),
        vec![77],
        vec![23; 70001],
        (0..65536).map(|x| x as u8).collect(),
    ] {
        fs::write(&input, &data).unwrap();
        for (mode, tag) in [
            (Codec::Huffman, 1),
            (Codec::Arithmetic, 2),
            (Codec::Ans, 3),
            (Codec::Auto, 0),
        ] {
            transcode(&input, &archive, Some(mode), 70001, &stop).unwrap();
            let packed = fs::read(&archive).unwrap();
            if !data.is_empty() && mode != Codec::Auto {
                assert_eq!(packed[12], tag);
            }
            if data.len() == 70001 {
                assert!(packed.len() < data.len());
            }
            if mode == Codec::Auto && data.len() == 65536 {
                assert_eq!(packed[12], 0);
            }
            transcode(&archive, &output, None, 70001, &stop).unwrap();
            assert_eq!(fs::read(&output).unwrap(), data);
            fs::remove_file(&archive).unwrap();
            fs::remove_file(&output).unwrap();
        }
    }
}

#[test]
fn corrupt_truncated_oversized_and_cancelled_never_publish() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input");
    let archive = dir.path().join("archive");
    let output = dir.path().join("out");
    fs::write(&input, vec![0u8; 8192]).unwrap();
    let stop = AtomicBool::new(false);
    transcode(&input, &archive, Some(Codec::Ans), 8192, &stop).unwrap();
    assert!(transcode(&archive, &output, None, 8191, &stop).is_err());
    assert!(transcode(&archive, &output, None, 8192, &AtomicBool::new(true)).is_err());
    let data = fs::read(&archive).unwrap();
    for index in [0, 12, 16, 30, data.len() - 1] {
        let mut changed = data.clone();
        changed[index] ^= 255;
        fs::write(&archive, changed).unwrap();
        assert!(transcode(&archive, &output, None, 8192, &stop).is_err());
        assert!(!output.exists());
    }
    for prefix in [0, 4, 12, data.len() - 1] {
        fs::write(&archive, &data[..prefix]).unwrap();
        assert!(transcode(&archive, &output, None, 8192, &stop).is_err());
    }
}
