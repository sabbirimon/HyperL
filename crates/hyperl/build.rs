// SPDX-License-Identifier: Apache-2.0
// HyperL Native Engine Cargo Build Script

fn main() {
    let mut build = cc::Build::new();
    build.file("../../native/engine.c")
         .file("../../native/kernels.c")
         .include("../../native")
         .include("../../native/include")
         .flag_if_supported("-O3")
         .flag_if_supported("-Wall")
         .flag_if_supported("-Wextra");

    #[cfg(target_arch = "x86_64")]
    {
        build.flag_if_supported("-mavx2");
        build.flag_if_supported("-mfma");
    }

    build.compile("hyperl_engine");

    println!("cargo:rerun-if-changed=../../native/engine.c");
    println!("cargo:rerun-if-changed=../../native/kernels.c");
    println!("cargo:rerun-if-changed=../../native/kernels.h");
    println!("cargo:rerun-if-changed=../../native/include/hyperl_engine.h");
}
