use anyhow::Result;

#[cfg(all(feature = "cuda", target_family = "unix"))]
const CUDA_NVCC_FLAGS: Option<&'static str> = option_env!("CUDA_NVCC_FLAGS");

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn main() -> Result<()> {
    use std::fs::OpenOptions;
    use std::io::prelude::*;
    use std::path::PathBuf;

    const OTHER_CONTENT: &str = r#"
pub const USE_FP8: bool = false;

pub(crate) mod ffi;
pub mod moe;

#[cfg(feature = "cuda")]
pub use mistralrs_paged_attn::{
    copy_blocks, kv_scale_update, paged_attention, reshape_and_cache, swap_blocks,
};
    "#;

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/cuda/moe_gemm.cu");
    println!("cargo:rerun-if-changed=src/cuda/moe_gemm_wmma.cu");
    println!("cargo:rerun-if-changed=src/cuda/moe_gemv.cu");
    println!("cargo:rerun-if-changed=src/cuda/sort.cu");
    // Detect CUDA compute capability for FP8 support
    let compute_cap = get_compute_cap();

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

    if compute_cap > 0 {
        // WMMA (Tensor Core) operations require SM 7.0+ (Volta)
        if compute_cap < 700 {
            builder = builder.arg("-DNO_WMMA_KERNEL");
        }
        // bf16 WMMA operations and intrinsics require SM 8.0+ (Ampere)
        if compute_cap < 800 {
            builder = builder.arg("-DNO_BF16_KERNEL");
        }
    }

    // Enable FP8 if compute capability >= 8.0 (Ampere and newer)
    let using_fp8 = if compute_cap >= 800 {
        builder = builder.arg("-DENABLE_FP8");
        true
    } else {
        false
    };

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
        build_dir.join("mistralrscorecuda.lib")
    } else {
        build_dir.join("libmistralrscorecuda.a")
    };
    builder.build_lib(out_file);

    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=mistralrscorecuda");
    println!("cargo:rustc-link-lib=dylib=cudart");

    let mut file = OpenOptions::new()
        .write(true)
        .open("src/cuda/mod.rs")
        .unwrap();

    // Build the new content
    let new_ct = if using_fp8 {
        &OTHER_CONTENT
            .trim()
            .replace("USE_FP8: bool = false", "USE_FP8: bool = true")
    } else {
        OTHER_CONTENT.trim()
    };

    // Add the other stuff back
    if let Err(e) = writeln!(file, "{new_ct}") {
        anyhow::bail!("Error while building dependencies: {:?}\n", e)
    }
    Ok(())
}

#[cfg(feature = "metal")]
fn main() -> Result<(), String> {
    use std::path::PathBuf;
    use std::process::Command;
    use std::{env, str};

    const METAL_SOURCES: [&str; 4] = [
        "copy_blocks",
        "pagedattention",
        "reshape_and_cache",
        "kv_scale_update",
    ];
    for src in METAL_SOURCES {
        println!("cargo::rerun-if-changed=src/metal/kernels/{src}.metal");
    }
    println!("cargo::rerun-if-changed=src/metal/kernels/utils.metal");
    println!("cargo::rerun-if-changed=src/metal/kernels/float8.metal");
    println!("cargo::rerun-if-changed=build.rs");

    // Check if precompilation should be skipped
    // https://github.com/EricLBuehler/mistral.rs/pull/1311#issuecomment-3001309885
    println!("cargo:rerun-if-env-changed=MISTRALRS_METAL_PRECOMPILE");
    let skip_precompile = env::var("MISTRALRS_METAL_PRECOMPILE")
        .map(|v| v == "0" || v.to_lowercase() == "false")
        .unwrap_or(false);

    if skip_precompile {
        println!(
            "cargo:warning=Skipping Metal kernel precompilation (MISTRALRS_METAL_PRECOMPILE=0)"
        );
        // Write a dummy metallib file to satisfy the include_bytes! macro
        let out_dir = PathBuf::from(std::env::var("OUT_DIR").map_err(|_| "OUT_DIR not set")?);
        std::fs::write(out_dir.join("mistralrs_paged_attention.metallib"), []).unwrap();
        std::fs::write(out_dir.join("mistralrs_paged_attention_ios.metallib"), []).unwrap();
        return Ok(());
    }

    enum Platform {
        MacOS,
        Ios,
    }

    impl Platform {
        fn sdk(&self) -> &str {
            match self {
                Platform::MacOS => "macosx",
                Platform::Ios => "iphoneos",
            }
        }
    }

    fn compile(platform: Platform) -> Result<(), String> {
        let current_dir = env::current_dir().expect("Failed to get current directory");
        let out_dir = PathBuf::from(std::env::var("OUT_DIR").map_err(|_| "OUT_DIR not set")?);
        let working_directory = out_dir.to_string_lossy().to_string();
        let sources = current_dir.join("src").join("metal").join("kernels");

        // Compile metal to air
        let mut compile_air_cmd = Command::new("xcrun");
        compile_air_cmd
            .arg("--sdk")
            .arg(platform.sdk())
            .arg("metal")
            .arg(format!("-working-directory={working_directory}"))
            .arg("-Wall")
            .arg("-Wextra")
            .arg("-O3")
            .arg("-c")
            .arg("-w");
        for metal_file in METAL_SOURCES {
            compile_air_cmd.arg(sources.join(format!("{metal_file}.metal")));
        }
        compile_air_cmd.arg(sources.join("utils.metal"));
        compile_air_cmd.arg(sources.join("float8.metal"));
        compile_air_cmd
            .spawn()
            .expect("Failed to compile air")
            .wait()
            .expect("Failed to compile air");

        let mut child = compile_air_cmd.spawn().expect("Failed to compile air");

        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    panic!("Compiling metal -> air failed. Exit with status: {status}")
                }
            }
            Ok(None) => {
                let status = child
                    .wait()
                    .expect("Compiling metal -> air failed while waiting for result");
                if !status.success() {
                    panic!("Compiling metal -> air failed. Exit with status: {status}")
                }
            }
            Err(e) => panic!("Compiling metal -> air failed: {e:?}"),
        }

        // Compile air to metallib
        let lib_name = match platform {
            Platform::MacOS => "mistralrs_paged_attention.metallib",
            Platform::Ios => "mistralrs_paged_attention_ios.metallib",
        };
        let metallib = out_dir.join(lib_name);
        let mut compile_metallib_cmd = Command::new("xcrun");
        compile_metallib_cmd.arg("metal").arg("-o").arg(&metallib);

        for metal_file in METAL_SOURCES {
            compile_metallib_cmd.arg(out_dir.join(format!("{metal_file}.air")));
        }
        compile_metallib_cmd.arg(out_dir.join("utils.air"));
        compile_metallib_cmd.arg(out_dir.join("float8.air"));

        let mut child = compile_metallib_cmd
            .spawn()
            .expect("Failed to compile air -> metallib");

        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    panic!("Compiling air -> metallib failed. Exit with status: {status}")
                }
            }
            Ok(None) => {
                let status = child
                    .wait()
                    .expect("Compiling air -> metallib failed while waiting for result");
                if !status.success() {
                    panic!("Compiling air -> metallib failed. Exit with status: {status}")
                }
            }
            Err(e) => panic!("Compiling air -> metallib failed: {e:?}"),
        }

        Ok(())
    }

    compile(Platform::MacOS)?;
    compile(Platform::Ios)?;

    Ok(())
}

#[cfg(not(any(all(feature = "cuda", target_family = "unix"), feature = "metal")))]
fn main() -> Result<()> {
    Ok(())
}

/// Get CUDA compute capability using cudarc driver detection.
/// Falls back to CUDA_COMPUTE_CAP env var if driver detection fails.
/// Returns the MINIMUM compute cap to ensure compatibility with all GPUs.
fn get_compute_cap() -> usize {
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");

        // First try to detect from actual GPU hardware via cudarc
        if let Ok(caps) = list_compute_caps() {
            if !caps.is_empty() {
                // Use minimum compute cap to ensure all kernels work on all GPUs
                // (for multi-GPU setups with different architectures)
                // TODO: support multiple compute caps (related to https://github.com/Narsil/bindgen_cuda/pull/16)
                let min_cap = *caps.iter().min().unwrap();
                println!(
                    "cargo:warning=Using detected compute cap: {} (from {:?})",
                    min_cap, caps
                );
                return min_cap as usize;
            }
        }

        // Fallback to environment variable
        if let Ok(compute_cap_str) = std::env::var("CUDA_COMPUTE_CAP") {
            if let Ok(cap) = compute_cap_str.parse::<usize>() {
                println!("cargo:warning=Using CUDA_COMPUTE_CAP from env: {}", cap);
                return cap;
            }
        }

        // Default to 0 if nothing worked - bindgen_cuda will try to detect it
        println!("cargo:warning=Could not detect compute cap, defaulting to 0");
    }
    0
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn list_compute_caps() -> Result<Vec<usize>, ()> {
    // Try to initialize the CUDA driver and query devices
    if candle_core::cuda::cudarc::driver::result::init().is_err() {
        println!("cargo:warning=CUDA driver init failed; falling back to nvidia-smi or env var");
        return Err(());
    }

    let n = candle_core::cuda::cudarc::driver::result::device::get_count()
        .map(|x| x as usize)
        .unwrap_or(0);

    let mut seen = std::collections::HashSet::new();
    let mut devices_cc = Vec::with_capacity(n);

    for i in 0..n {
        let ctx = match candle_core::cuda::cudarc::driver::CudaContext::new(i) {
            Ok(c) => c,
            Err(e) => {
                println!(
                    "cargo:warning=Failed to create CUDA context for device {}: {:?}",
                    i, e
                );
                continue;
            }
        };

        let cc = match ctx.compute_capability() {
            // Format: XY0 (e.g., 610 for SM 6.1, 700 for SM 7.0) - matches existing FP8 logic
            Ok((major, minor)) => (major as usize) * 100 + (minor as usize) * 10,
            Err(e) => {
                println!(
                    "cargo:warning=Failed to get compute cap for device {}: {:?}",
                    i, e
                );
                continue;
            }
        };

        println!(
            "cargo:warning=CUDA device id {} has compute capability {}",
            i, cc
        );

        if seen.insert(cc) {
            devices_cc.push(cc);
        }
    }

    if devices_cc.len() > 1 {
        println!(
            "cargo:warning=Multiple compute capabilities detected: {:?}",
            devices_cc
        );
    }

    devices_cc.sort_unstable();
    Ok(devices_cc)
}
