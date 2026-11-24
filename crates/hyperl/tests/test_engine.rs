// SPDX-License-Identifier: Apache-2.0

use hyperl::{version, Context, DeviceType, DType, Tensor, rmsnorm, softmax, gemv};

#[test]
fn test_version_and_context() {
    let ver = version();
    println!("HyperL Rust Engine Version: {}", ver);
    assert!(!ver.is_empty());

    let ctx = Context::new(DeviceType::Cpu, 4).expect("Failed to create Context");
    assert_eq!(ctx.threads(), 4);
}

#[test]
fn test_rmsnorm() {
    let ctx = Context::new(DeviceType::Cpu, 2).unwrap();
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x = Tensor::from_slice(&ctx, &[8], &data).unwrap();

    let normed = rmsnorm(&ctx, &x, None, 1e-5).unwrap();
    assert_eq!(normed.len(), 8);
    assert!((normed[0] - 0.1980295).abs() < 1e-4);
}

#[test]
fn test_gemv() {
    let ctx = Context::new(DeviceType::Cpu, 2).unwrap();
    let w_data = vec![1.0; 32]; // 4 x 8
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // sum = 36

    let w = Tensor::from_slice(&ctx, &[4, 8], &w_data).unwrap();
    let x = Tensor::from_slice(&ctx, &[8], &x_data).unwrap();

    let y = gemv(&ctx, &w, &x, None).unwrap();
    assert_eq!(y.len(), 4);
    for val in y.iter() {
        assert!((val - 36.0).abs() < 1e-4);
    }
}

#[test]
fn test_softmax() {
    let ctx = Context::new(DeviceType::Cpu, 1).unwrap();
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let x = Tensor::from_slice(&ctx, &[4], &data).unwrap();

    let s = softmax(&ctx, &x).unwrap();
    let sum: f32 = s.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}
