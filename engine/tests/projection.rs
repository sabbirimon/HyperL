// SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
use hyperl_engine::{Plan, Projection, Weights};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use sha2::{Digest, Sha256};
use std::{
    fs,
    process::Command,
    sync::atomic::{AtomicBool, Ordering},
};
use tempfile::tempdir;

fn fixture(dtype: Dtype, shape: Vec<usize>, weights: &[u8]) -> Vec<u8> {
    let x: Vec<u8> = [1f32, 2., 3., 4.]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let a = TensorView::new(Dtype::F32, vec![2, 2], &x).unwrap();
    let b = TensorView::new(dtype, shape, weights).unwrap();
    safetensors::tensor::serialize([("x", a), ("w", b)], None).unwrap()
}

#[test]
fn matrix_encodings_share_the_same_reference() {
    // x [[1,2],[3,4]], w [[1,-2],[3,4]].
    for (dtype, bytes, shape, scale) in [
        (
            Dtype::F32,
            [1f32, -2., 3., 4.]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            vec![2, 2],
            None,
        ),
        (
            Dtype::BF16,
            vec![128, 63, 0, 192, 64, 64, 128, 64],
            vec![2, 2],
            None,
        ),
        (
            Dtype::F8_E4M3,
            vec![0x38, 0xc0, 0x44, 0x48],
            vec![2, 2],
            None,
        ),
        (
            Dtype::F8_E5M2,
            vec![0x3c, 0xc0, 0x42, 0x44],
            vec![2, 2],
            None,
        ),
        (Dtype::U8, vec![0xe1, 0x43], vec![2, 1], Some(1.0)),
    ] {
        let bytes = fixture(dtype, shape, &bytes);
        let dir = tempdir().unwrap();
        let path = dir.path().join("model");
        fs::write(&path, &bytes).unwrap();
        let stop = AtomicBool::new(false);
        let file =
            unsafe { Weights::open(&path, &Sha256::digest(&bytes).into(), 4096, &stop) }.unwrap();
        let plan = Projection::prepare(&file, "x", "w", scale, bytes.len() + 16, None).unwrap();
        let answer: Vec<u8> = [-3f32, 11., -5., 25.]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(plan.execute(&stop).unwrap(), answer);
        assert!(Projection::prepare(&file, "x", "w", scale, bytes.len() + 15, None).is_err());
        stop.store(true, Ordering::Relaxed);
        assert!(plan.execute(&stop).is_err());
    }
}

#[test]
fn plan_respects_local_capacity_decode_width_and_overflow() {
    assert_eq!(
        Plan::for_projection(1, 16, 100, 164, 8, None)
            .unwrap()
            .workers,
        8
    );
    assert_eq!(
        Plan::for_projection(10, 16, 100, 740, 8, Some(2))
            .unwrap()
            .workers,
        2
    );
    assert!(Plan::for_projection(1, 1, 1, 32, 4, Some(5)).is_err());
    assert!(Plan::for_projection(usize::MAX, 2, 1, usize::MAX, 4, None).is_err());
}

#[test]
fn cli_publishes_only_complete_verified_outputs() {
    let dir = tempdir().unwrap();
    let raw = fixture(Dtype::U8, vec![2, 1], &[0xe1, 0x43]);
    let path = dir.path().join("weights");
    fs::write(&path, &raw).unwrap();
    let output = dir.path().join("result");
    let hash = format!("{:x}", Sha256::digest(&raw));
    let run = |hash: &str| {
        Command::new(env!("CARGO_BIN_EXE_hyperl-engine"))
            .args([
                "project",
                path.to_str().unwrap(),
                hash,
                "x",
                "w",
                "4096",
                "1",
                output.to_str().unwrap(),
                "--int4-scale",
                "1",
            ])
            .output()
            .unwrap()
    };
    assert!(!run(&"0".repeat(64)).status.success());
    assert!(!output.exists());
    assert!(run(&hash).status.success());
    let got = fs::read(&output).unwrap();
    let model = SafeTensors::deserialize(&got).unwrap();
    assert_eq!(model.tensor("output").unwrap().shape(), [2, 2]);
    assert!(!run(&hash).status.success());
    assert_eq!(got, fs::read(&output).unwrap());
}

#[test]
fn nonfinite_and_invalid_matrix_shapes_fail() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("f");
    for (shape, bytes) in [(vec![1, 2], vec![0x7f, 0x38]), (vec![1, 1], vec![0x38])] {
        let raw = fixture(Dtype::F8_E4M3, shape, &bytes);
        fs::write(&path, &raw).unwrap();
        let stop = AtomicBool::new(false);
        let file =
            unsafe { Weights::open(&path, &Sha256::digest(&raw).into(), 4096, &stop) }.unwrap();
        assert!(Projection::prepare(&file, "x", "w", None, 4096, None)
            .and_then(|p| p.execute(&stop))
            .is_err());
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires an explicitly selected physical GPU"]
fn physical_gpu_projection() {
    let adapter =
        std::env::var("HYPERL_GPU_ADAPTER").expect("set exact backend/name from processors");
    let m = 11usize;
    let n = 13usize;
    let k = 17usize;
    let x: Vec<f32> = (0..m * k).map(|i| ((i % 23) as f32 - 11.) / 7.).collect();
    let w: Vec<f32> = (0..n * k).map(|i| ((i % 19) as f32 - 9.) / 13.).collect();
    let xb: Vec<_> = x.iter().flat_map(|f| f.to_le_bytes()).collect();
    let wb: Vec<_> = w.iter().flat_map(|f| f.to_le_bytes()).collect();
    let raw = safetensors::tensor::serialize(
        [
            ("x", TensorView::new(Dtype::F32, vec![m, k], &xb).unwrap()),
            ("w", TensorView::new(Dtype::F32, vec![n, k], &wb).unwrap()),
        ],
        None,
    )
    .unwrap();
    let dir = tempdir().unwrap();
    let path = dir.path().join("model");
    fs::write(&path, &raw).unwrap();
    let stop = AtomicBool::new(false);
    let file =
        unsafe { Weights::open(&path, &Sha256::digest(&raw).into(), 1 << 20, &stop) }.unwrap();
    let p = Projection::prepare(&file, "x", "w", None, 1 << 20, None).unwrap();
    assert!(hyperl_engine::gpu::project(&p, &adapter, p.plan().tensor_bytes, &stop).is_err());
    let got = hyperl_engine::gpu::project(&p, &adapter, 1 << 20, &stop).unwrap();
    for (i, value) in got.chunks_exact(4).enumerate() {
        let expected: f64 = (0..k)
            .map(|t| f64::from(x[(i / n) * k + t]) * f64::from(w[(i % n) * k + t]))
            .sum();
        assert!((f64::from(f32::from_le_bytes(value.try_into().unwrap())) - expected).abs() < 1e-5);
    }
    println!("verified {adapter}, shape [{m},{k}] by [{n},{k}]");
}
