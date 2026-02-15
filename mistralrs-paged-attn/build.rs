use anyhow::Result;

#[cfg(all(feature = "cuda", target_family = "unix"))]
const CUDA_NVCC_FLAGS: Option<&'static str> = option_env!("CUDA_NVCC_FLAGS");

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn main() -> Result<()> {
    use std::path::PathBuf;

    // Declare expected cfg values for check-cfg lint
    println!("cargo::rustc-check-cfg=cfg(has_fp8)");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/cuda/pagedattention.cuh");
    println!("cargo:rerun-if-changed=src/cuda/copy_blocks_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/reshape_and_cache_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/concat_and_cache_mla_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/gather_mla_cache_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/gather_kv_cache_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer_mla_decode.cu");
    println!("cargo:rerun-if-changed=src/cuda/update_kvscales.cu");
    println!("cargo:rerun-if-changed=src/cuda/flash_attn_sinks.cu");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/cp_async.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/exception.h");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/fastdiv.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/layout.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/math.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/page.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/pos_enc.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/utils.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/vec_dtypes.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/attention/cascade.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/attention/decode.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/attention/default_decode_params.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/attention/state.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/attention/variant_helper.cuh");
    println!("cargo:rerun-if-changed=src/cuda/flashinfer/attention/variants.cuh");

    let mut builder = cudaforge::KernelBuilder::new()
        .source_glob("src/cuda/*.cu")
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

    let compute_cap = builder.get_compute_cap().unwrap_or(80);
    // Enable FP8 if compute capability >= 8.0 (Ampere and newer)
    let using_fp8 = if compute_cap >= 80 {
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
    builder
        .build_lib(out_file)
        .expect("Build paged attention lib failed!");

    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=mistralrspagedattention");
    println!("cargo:rustc-link-lib=dylib=cudart");

    if using_fp8 {
        println!("cargo:rustc-cfg=has_fp8");
    }
    Ok(())
}

#[cfg(feature = "metal")]
fn main() -> Result<(), String> {
    use std::path::PathBuf;
    use std::process::Command;
    use std::{env, str};

    // Declare expected cfg values for check-cfg lint
    println!("cargo::rustc-check-cfg=cfg(has_fp8)");

    const METAL_SOURCES: [&str; 5] = [
        "copy_blocks",
        "pagedattention",
        "reshape_and_cache",
        "kv_scale_update",
        "gather_kv_cache",
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
        std::fs::write(out_dir.join("mistralrs_paged_attention_tvos.metallib"), []).unwrap();
        return Ok(());
    }

    enum Platform {
        MacOS,
        Ios,
        TvOS,
    }

    impl Platform {
        fn sdk(&self) -> &str {
            match self {
                Platform::MacOS => "macosx",
                Platform::Ios => "iphoneos",
                Platform::TvOS => "appletvos",
            }
        }

        fn metal_std(&self) -> &str {
            // Use Metal 3.0 unified standard for all platforms.
            // This fixes Xcode 26+ where the default Metal standard may be too low.
            // https://github.com/EricLBuehler/mistral.rs/issues/1844
            //
            // Note: tvOS devices with A15+ (Apple TV 4K 3rd gen) support Metal 3.0+.
            match self {
                Platform::MacOS | Platform::Ios | Platform::TvOS => "metal3.0",
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
            .arg(format!("-std={}", platform.metal_std()))
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
            Platform::TvOS => "mistralrs_paged_attention_tvos.metallib",
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
    compile(Platform::TvOS)?;

    Ok(())
}

#[cfg(not(any(all(feature = "cuda", target_family = "unix"), feature = "metal")))]
fn main() -> Result<()> {
    // Declare expected cfg values for check-cfg lint
    println!("cargo::rustc-check-cfg=cfg(has_fp8)");
    Ok(())
}
