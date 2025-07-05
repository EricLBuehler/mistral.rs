use anyhow::Result;

#[cfg(all(feature = "cuda", target_family = "unix"))]
const CUDA_NVCC_FLAGS: Option<&'static str> = option_env!("CUDA_NVCC_FLAGS");

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn main() -> Result<()> {
    use std::fs::OpenOptions;
    use std::io::prelude::*;
    use std::path::PathBuf;
    use std::process::Command;

    const OTHER_CONTENT: &str = r#"
pub const USE_FP8: bool = false;

mod backend;
mod ffi;

pub use backend::{copy_blocks, paged_attention, reshape_and_cache, swap_blocks};
    "#;

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/cuda/pagedattention.cuh");
    println!("cargo:rerun-if-changed=src/cuda/copy_blocks_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/reshape_and_cache_kernel.cu");

    // Detect CUDA compute capability for FP8 support
    let compute_cap = {
        if let Ok(var) = std::env::var("CUDA_COMPUTE_CAP") {
            var.parse::<usize>().unwrap() * 10
        } else {
            let mut cmd = Command::new("nvidia-smi");
            match cmd
                .args(["--query-gpu=compute_cap", "--format=csv"])
                .output()
            {
                Ok(out) => {
                    let output =
                        String::from_utf8(out.stdout).expect("Output of nvidia-smi was not utf8.");
                    (output
                        .split('\n')
                        .nth(1)
                        .unwrap()
                        .trim()
                        .parse::<f32>()
                        .unwrap()
                        * 100.) as usize
                }
                Err(_) => {
                    // If nvidia-smi fails, assume no FP8 support
                    println!(
                        "cargo:warning=Could not detect CUDA compute capability, disabling FP8"
                    );
                    0
                }
            }
        }
    };

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
        build_dir.join("mistralrspagedattention.lib")
    } else {
        build_dir.join("libmistralrspagedattention.a")
    };
    builder.build_lib(out_file);

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

    const METAL_SOURCES: [&str; 3] = ["copy_blocks", "pagedattention", "reshape_and_cache"];
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
        std::fs::write(out_dir.join("mistralrs_paged_attention.metallib"), &[]).unwrap();
        std::fs::write(out_dir.join("mistralrs_paged_attention_ios.metallib"), &[]).unwrap();
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
                    panic!(
                        "Compiling metal -> air failed. Exit with status: {}",
                        status
                    )
                }
            }
            Ok(None) => {
                let status = child
                    .wait()
                    .expect("Compiling metal -> air failed while waiting for result");
                if !status.success() {
                    panic!(
                        "Compiling metal -> air failed. Exit with status: {}",
                        status
                    )
                }
            }
            Err(e) => panic!("Compiling metal -> air failed: {:?}", e),
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
                    panic!(
                        "Compiling air -> metallib failed. Exit with status: {}",
                        status
                    )
                }
            }
            Ok(None) => {
                let status = child
                    .wait()
                    .expect("Compiling air -> metallib failed while waiting for result");
                if !status.success() {
                    panic!(
                        "Compiling air -> metallib failed. Exit with status: {}",
                        status
                    )
                }
            }
            Err(e) => panic!("Compiling air -> metallib failed: {:?}", e),
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
