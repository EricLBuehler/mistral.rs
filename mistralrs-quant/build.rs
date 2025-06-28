fn main() -> Result<(), String> {
    #[cfg(feature = "cuda")]
    {
        use std::{fs::read_to_string, path::PathBuf, process::Command, vec};
        const MARLIN_FFI_PATH: &str = "src/gptq/marlin_ffi.rs";
        const BLOCKWISE_FP8_FFI_PATH: &str = "src/blockwise_fp8/ffi.rs";
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

        // ======== Handle optional marlin kernel compilation
        let cc_over_800 = compute_cap >= 800;
        let cc_is_over_800 = match cc_over_800 {
            true => "true",
            false => "false",
        };

        let mut marlin_ffi_ct = read_to_string(MARLIN_FFI_PATH).unwrap();
        if marlin_ffi_ct.contains("pub(crate) const HAVE_MARLIN_KERNELS: bool = true;") {
            marlin_ffi_ct = marlin_ffi_ct.replace(
                "pub(crate) const HAVE_MARLIN_KERNELS: bool = true;",
                &format!("pub(crate) const HAVE_MARLIN_KERNELS: bool = {cc_is_over_800};"),
            );
        } else {
            marlin_ffi_ct = marlin_ffi_ct.replace(
                "pub(crate) const HAVE_MARLIN_KERNELS: bool = false;",
                &format!("pub(crate) const HAVE_MARLIN_KERNELS: bool = {cc_is_over_800};"),
            );
        }
        std::fs::write(MARLIN_FFI_PATH, marlin_ffi_ct).unwrap();

        let mut blockwise_fp8_ffi_ct = read_to_string(BLOCKWISE_FP8_FFI_PATH).unwrap();
        if blockwise_fp8_ffi_ct
            .contains("pub(crate) const HAVE_BLOCKWISE_DEQUANT_KERNELS: bool = true;")
        {
            blockwise_fp8_ffi_ct = blockwise_fp8_ffi_ct.replace(
                "pub(crate) const HAVE_BLOCKWISE_DEQUANT_KERNELS: bool = true;",
                &format!(
                    "pub(crate) const HAVE_BLOCKWISE_DEQUANT_KERNELS: bool = {cc_is_over_800};"
                ),
            );
        } else {
            blockwise_fp8_ffi_ct = blockwise_fp8_ffi_ct.replace(
                "pub(crate) const HAVE_BLOCKWISE_DEQUANT_KERNELS: bool = false;",
                &format!(
                    "pub(crate) const HAVE_BLOCKWISE_DEQUANT_KERNELS: bool = {cc_is_over_800};"
                ),
            );
        }
        std::fs::write(BLOCKWISE_FP8_FFI_PATH, blockwise_fp8_ffi_ct).unwrap();
        // ========

        let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
        let mut lib_files = vec![
            "kernels/gptq/q_gemm.cu",
            "kernels/hqq/hqq.cu",
            "kernels/ops/ops.cu",
            "kernels/bitsandbytes/dequant.cu",
            "kernels/rotary/rotary.cu",
        ];
        if cc_over_800 {
            lib_files.push("kernels/marlin/marlin_matmul_f16.cu");
            lib_files.push("kernels/marlin/marlin_matmul_bf16.cu");
            lib_files.push("kernels/marlin/marlin_matmul_awq_f16.cu");
            lib_files.push("kernels/marlin/marlin_matmul_awq_bf16.cu");
            lib_files.push("kernels/marlin/marlin_repack.cu");
            lib_files.push("kernels/blockwise_fp8/blockwise_fp8.cu");
        } else {
            lib_files.push("kernels/marlin/dummy_marlin_kernel.cu");
            lib_files.push("kernels/blockwise_fp8/blockwise_fp8_dummy.cu");
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

        Ok(())
    }

    #[cfg(feature = "metal")]
    {
        use std::path::PathBuf;
        use std::process::Command;
        use std::{env, str};

        const METAL_SOURCES: [&str; 8] = [
            "bitwise",
            "blockwise_fp8",
            "bnb_dequantize",
            "hqq_dequantize",
            "quantized",
            "scan",
            "sort",
            "copy",
        ];
        const HEADER_SOURCES: [&str; 5] = ["utils", "bf16", "scan_impl", "sort_impl", "copy_impl"];
        for src in METAL_SOURCES {
            println!("cargo::rerun-if-changed=src/metal_kernels/{src}.metal");
        }
        for src in HEADER_SOURCES {
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
            std::fs::write(out_dir.join("mistralrs_quant.metallib"), &[]).unwrap();
            std::fs::write(out_dir.join("mistralrs_quant_ios.metallib"), &[]).unwrap();
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
            let sources = current_dir.join("src").join("metal_kernels");

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

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    Ok(())
}
