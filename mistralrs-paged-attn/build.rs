use anyhow::Result;

#[cfg(all(feature = "cuda", target_family = "unix"))]
const CUDA_NVCC_FLAGS: Option<&'static str> = option_env!("CUDA_NVCC_FLAGS");

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn main() -> Result<()> {
    use std::fs::OpenOptions;
    use std::io::prelude::*;
    use std::path::PathBuf;

    const OTHER_CONTENT: &str = r#"
pub const COPY_BLOCKS_KERNEL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/copy_blocks_kernel.ptx"));
pub const PAGEDATTENTION: &str = include_str!(concat!(env!("OUT_DIR"), "/pagedattention.ptx"));
pub const RESHAPE_AND_CACHE_KERNEL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/reshape_and_cache_kernel.ptx"));

mod backend;
mod ffi;

pub use backend::{copy_blocks, paged_attention, reshape_and_cache, swap_blocks};
    "#;

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/cuda/pagedattention.cu");
    println!("cargo:rerun-if-changed=src/cuda/copy_blocks_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/reshape_and_cache_kernel.cu");
    let mut builder = bindgen_cuda::Builder::default()
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_HALF2_OPERATORS__")
        .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("--use_fast_math")
        .arg("--verbose")
        .arg("--compiler-options")
        .arg("-fPIC");

    // https://github.com/EricLBuehler/mistral.rs/issues/286
    if let Some(cuda_nvcc_flags_env) = CUDA_NVCC_FLAGS {
        builder = builder.arg("--compiler-options");
        builder = builder.arg(cuda_nvcc_flags_env);
    }
    println!("cargo:info={builder:?}");

    let target = std::env::var("TARGET").unwrap();
    let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    // https://github.com/EricLBuehler/mistral.rs/issues/588
    let out_file = if target.contains("msvc") {
        // Windows case
        build_dir.join("mistralrspagedattention.lib")
    } else {
        build_dir.join("libmistralrspagedattention.a")
    };
    builder.build_lib(out_file);

    let bindings = builder.build_ptx().unwrap();
    bindings.write("src/cuda/mod.rs").unwrap();

    let kernel_dir = PathBuf::from("../mistralrs-paged-attn");
    let absolute_kernel_dir = std::fs::canonicalize(kernel_dir).unwrap();

    println!(
        "cargo:rustc-link-search=native={}",
        absolute_kernel_dir.display()
    );
    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=mistralrspagedattention");
    println!("cargo:rustc-link-lib=dylib=cudart");

    let mut file = OpenOptions::new()
        .write(true)
        .open("src/cuda/mod.rs")
        .unwrap();

    // Add the other stuff back
    if let Err(e) = writeln!(file, "{}", OTHER_CONTENT.trim()) {
        anyhow::bail!("Error while building dependencies: {:?}\n", e)
    }
    Ok(())
}

#[cfg(not(all(feature = "cuda", target_family = "unix")))]
fn main() -> Result<()> {
    Ok(())
}
