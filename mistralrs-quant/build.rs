#[cfg(feature = "cuda")]
const CUTLASS_COMMIT: &str = "7127592069c2fe01b041e174ba4345ef9b279671";
#[cfg(feature = "cuda")]
const DEEPGEMM_CUTLASS_COMMIT: &str = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8";
#[cfg(feature = "cuda")]
const CUTLASS_COMMIT_ENV: &str = "MISTRALRS_CUTLASS_COMMIT";
#[cfg(feature = "cuda")]
const SUPPORTED_CUDA_TOOLKIT_VERSIONS: &[(usize, usize)] = &[
    (13, 3),
    (13, 2),
    (13, 1),
    (13, 0),
    (12, 9),
    (12, 8),
    (12, 6),
    (12, 5),
    (12, 4),
    (12, 3),
    (12, 2),
    (12, 1),
    (12, 0),
    (11, 8),
    (11, 7),
    (11, 6),
    (11, 5),
    (11, 4),
];

#[cfg(feature = "cuda")]
fn cuda_kernel_builder(build_dir: &std::path::Path) -> cudaforge::KernelBuilder {
    cudaforge::KernelBuilder::new()
        .out_dir(build_dir)
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
        .arg("-fPIC")
}

#[cfg(feature = "metal")]
include!("src/metal_kernels/source_set.rs");

#[cfg(feature = "cuda")]
#[path = "src/build_support/cuda_headers.rs"]
mod cuda_headers;

#[cfg(feature = "cuda")]
#[path = "src/build_support/deepgemm_jit_bundle.rs"]
mod deepgemm_jit_bundle;

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
    let (major, minor) = parse_cuda_version_from_nvcc(&stdout).unwrap_or_else(|| {
        panic!("Unsupported cuda toolkit version from `nvcc --version`:\n{stdout}")
    });
    if SUPPORTED_CUDA_TOOLKIT_VERSIONS.contains(&(major, minor)) {
        (major, minor)
    } else {
        panic!("Unsupported cuda toolkit version: `{major}.{minor}`. Please raise a github issue.")
    }
}

#[cfg(feature = "cuda")]
fn parse_cuda_version_from_nvcc(stdout: &str) -> Option<(usize, usize)> {
    let release = stdout.split("release ").nth(1)?;
    let version = release
        .split(|c: char| c == ',' || c.is_whitespace())
        .next()?;
    let mut parts = version.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().ok()?;
    Some((major, minor))
}

#[cfg(feature = "cuda")]
fn cutile_supported_for_build_cuda(major: usize, minor: usize, compute_cap: usize) -> bool {
    let cuda_code = major * 100 + minor;
    ((80..90).contains(&compute_cap) && cuda_code >= 1302)
        || (compute_cap == 90 && cuda_code >= 1303)
        || (compute_cap >= 100 && cuda_code >= 1302)
}

fn main() -> Result<(), String> {
    // Declare expected cfg values for check-cfg lint
    println!("cargo::rustc-check-cfg=cfg(has_marlin_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_blockwise_fp8_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_cutlass_fp8_sm90_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_deepgemm_fp8_sm90_provider)");
    println!("cargo::rustc-check-cfg=cfg(has_scalar_fp8_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_vector_fp8_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_mxfp4_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_mxfp4_wmma_kernels)");
    println!("cargo::rustc-check-cfg=cfg(has_cutlass_moe_kernels)");
    println!("cargo::rustc-check-cfg=cfg(cuda_ge_13000)");

    #[cfg(feature = "cuda")]
    {
        use std::{path::Path, path::PathBuf, vec};
        const CUDA_NVCC_FLAGS: Option<&'static str> = option_env!("CUDA_NVCC_FLAGS");

        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rerun-if-env-changed=CUDA_NVCC_FLAGS");
        println!("cargo:rerun-if-env-changed={CUTLASS_COMMIT_ENV}");
        let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
        let (cuda_major, cuda_minor) = cuda_version_from_build_system();

        let deepgemm_headers = cuda_headers::find(Path::new("third_party/deepgemm_sm90"))
            .map_err(|e| e.to_string())?;
        let deepgemm_generator_hash =
            cuda_headers::hash(&deepgemm_headers).map_err(|e| e.to_string())?;
        let header_files = cuda_headers::find(Path::new("kernels")).map_err(|e| e.to_string())?;
        for header_file in &header_files {
            println!("cargo:rerun-if-changed={}", header_file.display());
        }
        for header_file in &deepgemm_headers {
            println!("cargo:rerun-if-changed={}", header_file.display());
        }
        let header_hash_arg = format!(
            "-DMISTRALRS_QUANT_CUDA_HEADER_HASH=0x{:016x}",
            cuda_headers::hash(&header_files).map_err(|e| e.to_string())?
        );
        let kernel_build_dir = build_dir.join("kernels");
        let mut builder = cuda_kernel_builder(&kernel_build_dir)
            .source_glob("kernels/*/*.cu")
            .watch(["kernels"])
            .arg(&header_hash_arg);

        let compute_cap = builder.get_compute_cap().unwrap_or(80);
        // ======== Handle optional kernel compilation via rustc-cfg flags
        let cc_over_80 = compute_cap >= 80;
        let target = std::env::var("TARGET").unwrap();

        // The CUTLASS 2.x grouped-GEMM MoE kernels include gemm_universal.hpp, which textually pulls in
        // CUTLASS's Sm90 warp-specialized headers; cl (MSVC) fails to parse their dependent SharedStorage
        // types. Skip them on MSVC and let the engine fall back to the Fused MoE backend.
        let cutlass_moe = cc_over_80 && !target.contains("msvc");

        if cc_over_80 {
            println!("cargo:rustc-cfg=has_marlin_kernels");
            println!("cargo:rustc-cfg=has_blockwise_fp8_kernels");
            println!("cargo:rustc-cfg=has_scalar_fp8_kernels");
            println!("cargo:rustc-cfg=has_vector_fp8_kernels");
            // WMMA tensor core MXFP4 kernel (FP16/BF16 WMMA requires SM >= 80)
            println!("cargo:rustc-cfg=has_mxfp4_wmma_kernels");
        }
        let cutlass_fp8_sm90 = compute_cap == 90 && cuda_major >= 12 && !target.contains("msvc");
        if cutlass_fp8_sm90 {
            println!("cargo:rustc-cfg=has_cutlass_fp8_sm90_kernels");
        }
        let deepgemm_fp8_sm90 = compute_cap == 90
            && (cuda_major > 12 || (cuda_major == 12 && cuda_minor >= 8))
            && target.contains("linux");
        let mut deepgemm_source_hash = deepgemm_generator_hash;
        if deepgemm_fp8_sm90 {
            println!("cargo:rustc-cfg=has_deepgemm_fp8_sm90_provider");
            let deepgemm_cutlass =
                cudaforge::ExternalDependency::cutlass(Some(DEEPGEMM_CUTLASS_COMMIT))
                    .fetch(&build_dir)
                    .map_err(|error| error.to_string())?;
            deepgemm_source_hash = deepgemm_jit_bundle::write(
                &build_dir.join("deepgemm_jit_headers.rs"),
                Path::new("third_party/deepgemm_sm90/include/official"),
                Path::new("third_party/deepgemm_sm90/include/deep_gemm"),
                &deepgemm_cutlass.join("include"),
                deepgemm_generator_hash,
            )
            .map_err(|error| error.to_string())?;
        }
        println!("cargo:rustc-env=MISTRALRS_DEEPGEMM_SOURCE_HASH={deepgemm_source_hash:016x}");
        // CUTLASS grouped-GEMM MoE fallback (Sm80 tensor-op, runs on sm_80+)
        if cutlass_moe {
            println!("cargo:rustc-cfg=has_cutlass_moe_kernels");
        }
        // MXFP4 is always enabled with CUDA (uses LUT-based dequantization)
        println!("cargo:rustc-cfg=has_mxfp4_kernels");

        let mut excluded_files = if cc_over_80 {
            vec!["dummy_*.cu", "*_dummy.cu"]
        } else {
            vec![
                "marlin_*.cu",
                "*_fp8.cu",
                "*_fp8_gemm.cu",
                "*_wmma.cu",
                "moe_data.cu",
                "grouped_mm_*.cu",
            ]
        };
        if cc_over_80 && !cutlass_moe {
            excluded_files.push("moe_data.cu");
            excluded_files.push("grouped_mm_*.cu");
        }
        if !cutlass_fp8_sm90 {
            excluded_files.push("*_cutlass_sm90.cu");
        }
        builder = builder.exclude(&excluded_files);
        let cutlass_commit =
            std::env::var(CUTLASS_COMMIT_ENV).unwrap_or_else(|_| CUTLASS_COMMIT.to_string());
        builder = builder.with_cutlass(Some(&cutlass_commit));

        // https://github.com/EricLBuehler/mistral.rs/issues/286
        if let Some(cuda_nvcc_flags_env) = CUDA_NVCC_FLAGS {
            builder = builder.arg("--compiler-options");
            builder = builder.arg(cuda_nvcc_flags_env);
        }

        let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

        // CUDA 13.x CCCL headers require MSVC's conforming preprocessor.
        if target.contains("msvc") {
            builder = builder.arg("--compiler-options").arg("/Zc:preprocessor");
        }

        // https://github.com/EricLBuehler/mistral.rs/issues/588
        let out_file = if target.contains("msvc") {
            // Windows case
            build_dir.join("mistralrsquant.lib")
        } else {
            build_dir.join("libmistralrsquant.a")
        };
        builder
            .build_lib(out_file)
            .expect("Build mistral quant lib failed!");
        if deepgemm_fp8_sm90 {
            let deepgemm_source_hash_arg =
                format!("-DMISTRALRS_DEEPGEMM_SOURCE_HASH=0x{deepgemm_source_hash:016x}");
            let deepgemm_build_dir = build_dir.join("deepgemm");
            let mut deepgemm_builder = cuda_kernel_builder(&deepgemm_build_dir)
                .source_files(["third_party/deepgemm_sm90/mistralrs_deepgemm_sm90.cu"])
                .watch(["third_party/deepgemm_sm90"])
                .compute_cap_arch("90a")
                .arg(&deepgemm_source_hash_arg)
                .with_cutlass(Some(&cutlass_commit))
                .include_path("third_party/deepgemm_sm90")
                .include_path("third_party/deepgemm_sm90/include");
            if let Some(cuda_nvcc_flags_env) = CUDA_NVCC_FLAGS {
                deepgemm_builder = deepgemm_builder
                    .arg("--compiler-options")
                    .arg(cuda_nvcc_flags_env);
            }
            deepgemm_builder
                .build_lib(build_dir.join("libmistralrsdeepgemm.a"))
                .expect("Build mistral DeepGEMM provider failed!");
        }
        println!("cargo:rustc-link-search={}", build_dir.display());
        println!("cargo:rustc-link-lib=mistralrsquant");
        println!("cargo:rustc-link-lib=dylib=cudart");
        if deepgemm_fp8_sm90 {
            println!("cargo:rustc-link-lib=mistralrsdeepgemm");
            println!("cargo:rustc-link-lib=dylib=cuda");
        }

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

        let (major, minor) = (cuda_major, cuda_minor);
        println!("cargo:rustc-env=MISTRALRS_BUILD_CUDA_VERSION={major}.{minor}");
        println!(
            "cargo:rustc-env=MISTRALRS_BUILD_CUDA_VERSION_CODE={}",
            major * 100 + minor
        );
        println!("cargo:rustc-cfg=feature=\"cuda-{major}0{minor}0\"");
        if major >= 13 {
            println!("cargo:rustc-cfg=cuda_ge_13000");
        }

        let cuda_ge_132 = major > 13 || (major == 13 && minor >= 2);
        let cutile_supported = cutile_supported_for_build_cuda(major, minor, compute_cap);
        if std::env::var("CARGO_FEATURE_CUTILE").is_ok() {
            if !cuda_ge_132 {
                panic!(
                    "the `cutile` feature requires CUDA >= 13.2 to build (found {major}.{minor}); \
                     build without `--features cutile`"
                );
            } else if !cutile_supported {
                println!(
                    "cargo:warning=the `cutile` feature is enabled, but CUDA {major}.{minor} does \
                     not support cuTile for sm_{compute_cap}; runtime will use another MoE backend."
                );
            }
        } else if cutile_supported {
            println!(
                "cargo:warning=CUDA {major}.{minor} detected: enable the `cutile` feature for \
                 optimized kernels."
            );
        }

        Ok(())
    }

    #[cfg(feature = "metal")]
    {
        mistralrs_metal_compile::compile_metallibs(&QUANT_METAL_SOURCE_SET)
    }

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    Ok(())
}
