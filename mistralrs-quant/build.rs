#[cfg(feature = "cuda")]
#[allow(unused)]
fn cuda_version_from_build_system() -> (usize, usize) {
    let output = std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .expect("Failed to execute `nvcc`");

    if !output.status.success() {
        panic!(
            "`nvcc --version` failed.\nstdout:\n{}\n\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let version_line = stdout.lines().nth(3).unwrap();
    let release_section = version_line.split(", ").nth(1).unwrap();
    let version_number = release_section.split(' ').nth(1).unwrap();

    match version_number {
        "13.1" => (13, 1),
        "13.0" => (13, 0),
        "12.9" => (12, 9),
        "12.8" => (12, 8),
        "12.6" => (12, 6),
        "12.5" => (12, 5),
        "12.4" => (12, 4),
        "12.3" => (12, 3),
        "12.2" => (12, 2),
        "12.1" => (12, 1),
        "12.0" => (12, 0),
        "11.8" => (11, 8),
        "11.7" => (11, 7),
        "11.6" => (11, 6),
        "11.5" => (11, 5),
        "11.4" => (11, 4),
        v => panic!("Unsupported cuda toolkit version: `{v}`. Please raise a github issue."),
    }
}

fn main() -> Result<(), String> {
    // Declare expected cfg values for check-cfg lint
    println!("cargo::rustc-check-cfg=cfg(has_marlin_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_blockwise_fp8_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_scalar_fp8_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_vector_fp8_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_mxfp4_kernels)");

    #[cfg(feature = "cuda")]
    {
        use std::{path::PathBuf, process::Command, vec};
        const CUDA_NVCC_FLAGS: Option<&'static str> = option_env!("CUDA_NVCC_FLAGS");

        println!("cargo:rerun-if-changed=build.rs");

        // Try CUDA_COMPUTE_CAP then nvidia-smi
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
                        let output = String::from_utf8(out.stdout)
                            .expect("Output of nvidia-smi was not utf8.");
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
                        panic!("`CUDA_COMPUTE_CAP` env var not specified and `nvidia-smi` was not found.");
                    }
                }
            }
        };

        // ======== Handle optional kernel compilation via rustc-cfg flags
        let cc_over_800 = compute_cap >= 800;

        if cc_over_800 {
            println!("cargo:rustc-cfg=has_marlin_kernels");
            println!("cargo:rustc-cfg=has_blockwise_fp8_kernels");
            println!("cargo:rustc-cfg=has_scalar_fp8_kernels");
            println!("cargo:rustc-cfg=has_vector_fp8_kernels");
        }
        // MXFP4 is always enabled with CUDA (uses LUT-based dequantization)
        println!("cargo:rustc-cfg=has_mxfp4_kernels");
        // ========

        let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
        let mut lib_files = vec![
            "kernels/gptq/q_gemm.cu",
            "kernels/hqq/hqq.cu",
            "kernels/hqq/hqq_bitpack.cu",
            "kernels/ops/ops.cu",
            "kernels/bitsandbytes/dequant.cu",
            "kernels/rotary/rotary.cu",
            "kernels/afq/afq.cu",
            "kernels/afq/afq_gemm.cu",
            "kernels/mxfp4/mxfp4_gemm.cu", // MXFP4 works on all compute caps
            "kernels/gemv/gemv.cu",        // Custom GEMV for decode-phase inference
            "kernels/indexed_moe/indexed_moe.cu", // Indexed MoE forward for GGUF quantized weights
        ];
        if cc_over_800 {
            lib_files.push("kernels/marlin/marlin_matmul_f16.cu");
            lib_files.push("kernels/marlin/marlin_matmul_bf16.cu");
            lib_files.push("kernels/marlin/marlin_matmul_awq_f16.cu");
            lib_files.push("kernels/marlin/marlin_matmul_awq_bf16.cu");
            lib_files.push("kernels/marlin/marlin_repack.cu");
            lib_files.push("kernels/blockwise_fp8/blockwise_fp8.cu");
            lib_files.push("kernels/blockwise_fp8/blockwise_fp8_gemm.cu");
            lib_files.push("kernels/scalar_fp8/scalar_fp8.cu");
            lib_files.push("kernels/vector_fp8/vector_fp8.cu");
        } else {
            lib_files.push("kernels/marlin/dummy_marlin_kernel.cu");
            lib_files.push("kernels/blockwise_fp8/blockwise_fp8_dummy.cu");
            lib_files.push("kernels/blockwise_fp8/blockwise_fp8_gemm_dummy.cu");
            lib_files.push("kernels/scalar_fp8/scalar_fp8_dummy.cu");
            lib_files.push("kernels/vector_fp8/vector_fp8_dummy.cu");
        }
        for lib_file in lib_files.iter() {
            println!("cargo:rerun-if-changed={lib_file}");
        }
        let mut builder = bindgen_cuda::Builder::default()
            .kernel_paths(lib_files)
            .out_dir(build_dir.clone())
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

        let target = std::env::var("TARGET").unwrap();
        let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
        // https://github.com/EricLBuehler/mistral.rs/issues/588
        let out_file = if target.contains("msvc") {
            // Windows case
            build_dir.join("mistralrsquant.lib")
        } else {
            build_dir.join("libmistralrsquant.a")
        };
        builder.build_lib(out_file);
        println!("cargo:rustc-link-search={}", build_dir.display());
        println!("cargo:rustc-link-lib=mistralrsquant");
        println!("cargo:rustc-link-lib=dylib=cudart");

        if target.contains("msvc") {
            // nothing to link to
        } else if target.contains("apple")
            || target.contains("freebsd")
            || target.contains("openbsd")
        {
            println!("cargo:rustc-link-lib=dylib=c++");
        } else if target.contains("android") {
            println!("cargo:rustc-link-lib=dylib=c++_shared");
        } else {
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }

        let (major, minor) = cuda_version_from_build_system();
        println!("cargo:rustc-cfg=feature=\"cuda-{major}0{minor}0\"");

        Ok(())
    }

    #[cfg(feature = "metal")]
    {
        use std::path::PathBuf;
        use std::process::Command;
        use std::{env, str};

        const METAL_SOURCES: [&str; 12] = [
            "bitwise",
            "blockwise_fp8",
            "bnb_dequantize",
            "fused_glu",
            "hqq_dequantize",
            "hqq_bitpack",
            "mxfp4",
            "quantized",
            "scalar_fp8",
            "scan",
            "sort",
            "copy",
        ];
        const HEADER_SOURCES: [&str; 5] = ["utils", "bf16", "scan_impl", "sort_impl", "copy_impl"];
        // Include-only headers (not compiled directly, just tracked for changes)
        const INCLUDE_ONLY: [&str; 2] = ["float8", "float4"];
        for src in METAL_SOURCES {
            println!("cargo::rerun-if-changed=src/metal_kernels/{src}.metal");
        }
        for src in HEADER_SOURCES {
            println!("cargo::rerun-if-changed=src/metal_kernels/{src}.metal");
        }
        for src in INCLUDE_ONLY {
            println!("cargo::rerun-if-changed=src/metal_kernels/{src}.metal");
        }
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
            std::fs::write(out_dir.join("mistralrs_quant.metallib"), []).unwrap();
            std::fs::write(out_dir.join("mistralrs_quant_ios.metallib"), []).unwrap();
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

            fn metal_std(&self) -> &str {
                // Use Metal 3.0 unified standard for both platforms
                // This fixes Xcode 26+ where the default Metal standard may be too low
                // https://github.com/EricLBuehler/mistral.rs/issues/1844
                match self {
                    Platform::MacOS | Platform::Ios => "metal3.0",
                }
            }
        }

        fn compile(platform: Platform) -> Result<(), String> {
            let current_dir = env::current_dir().expect("Failed to get current directory");
            let out_dir = PathBuf::from(std::env::var("OUT_DIR").map_err(|_| "OUT_DIR not set")?);
            let working_directory = out_dir.to_string_lossy().to_string();
            let sources = current_dir.join("src").join("metal_kernels");

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
            for metal_file in HEADER_SOURCES {
                compile_air_cmd.arg(sources.join(format!("{metal_file}.metal")));
            }
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
                Platform::MacOS => "mistralrs_quant.metallib",
                Platform::Ios => "mistralrs_quant_ios.metallib",
            };
            let metallib = out_dir.join(lib_name);
            let mut compile_metallib_cmd = Command::new("xcrun");
            compile_metallib_cmd.arg("metal").arg("-o").arg(&metallib);

            for metal_file in METAL_SOURCES {
                compile_metallib_cmd.arg(out_dir.join(format!("{metal_file}.air")));
            }
            for metal_file in HEADER_SOURCES {
                compile_metallib_cmd.arg(out_dir.join(format!("{metal_file}.air")));
            }

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

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    Ok(())
}
