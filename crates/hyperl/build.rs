// SPDX-License-Identifier: Apache-2.0
// HyperL Native Engine Cargo Build Script

fn main() {
    let mut build = cc::Build::new();
    build.file("../../native/engine.c")
         .file("../../native/kernels.c")
         .define("HYPERL_STATIC", None)
         .include("../../native")
         .include("../../native/include")
         .flag_if_supported("-O3")
         .flag_if_supported("-Wall")
         .flag_if_supported("-Wextra");

    build.compile("hyperl_engine");

    println!("cargo:rerun-if-changed=../../native/engine.c");
    println!("cargo:rerun-if-changed=../../native/kernels.c");
    println!("cargo:rerun-if-changed=../../native/kernels.h");
    println!("cargo:rerun-if-changed=../../native/include/hyperl_engine.h");
}
