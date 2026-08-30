#[cfg(feature = "cuda")]
const CUDA_NVCC_FLAGS: Option<&'static str> = option_env!("CUDA_NVCC_FLAGS");
#[cfg(feature = "cuda")]
const FLASHINFER_GDN_COMMIT: &str = "28406af5b9134757acbd6bc44647fd00261d163f";
#[cfg(feature = "cuda")]
const CUTLASS_COMMIT: &str = "7127592069c2fe01b041e174ba4345ef9b279671";
#[cfg(feature = "cuda")]
const FLASHINFER_GDN_MIN_CUDA: u32 = 1208;
#[cfg(feature = "cuda")]
const GDN_FP8_PRODUCER_MIN_CUDA: u32 = 1108;
#[cfg(feature = "cuda")]
const CUDA_BUILD_ROOT_ENV: &str = "MISTRALRS_CUDA_BUILD_ROOT";

#[cfg(feature = "cuda")]
fn cuda_build_dir(out_dir: &std::path::Path, component: &str) -> std::path::PathBuf {
    println!("cargo:rerun-if-env-changed={CUDA_BUILD_ROOT_ENV}");
    let Some(root) = std::env::var_os(CUDA_BUILD_ROOT_ENV) else {
        return out_dir.to_path_buf();
    };
    let build_dir = std::path::PathBuf::from(root).join("core").join(component);
    std::fs::create_dir_all(&build_dir).expect("failed to create shared CUDA build directory");
    build_dir
}

#[cfg(feature = "cuda")]
fn cuda_header_hash(dir: &std::path::Path) -> std::io::Result<u64> {
    fn update(hash: &mut u64, bytes: &[u8]) {
        for byte in bytes {
            *hash ^= u64::from(*byte);
            *hash = hash.wrapping_mul(0x100000001b3);
        }
    }

    fn visit(path: &std::path::Path, hash: &mut u64) -> std::io::Result<()> {
        if path.is_dir() {
            let mut entries = std::fs::read_dir(path)?
                .map(|entry| entry.map(|entry| entry.path()))
                .collect::<std::io::Result<Vec<_>>>()?;
            entries.sort();
            for entry in entries {
                visit(&entry, hash)?;
            }
            return Ok(());
        }

        let Some(extension) = path.extension().and_then(|extension| extension.to_str()) else {
            return Ok(());
        };
        if !matches!(extension, "h" | "cuh" | "hpp") {
            return Ok(());
        }

        update(hash, path.to_string_lossy().as_bytes());
        update(hash, &std::fs::read(path)?);
        Ok(())
    }

    let mut hash = 0xcbf29ce484222325;
    visit(dir, &mut hash)?;
    Ok(hash)
}

fn main() {
    set_git_revision();

    println!("cargo::rustc-check-cfg=cfg(has_flashinfer_gdn_sm90_kernel)");
    println!("cargo::rustc-check-cfg=cfg(has_gdn_fp8_producer)");

    #[cfg(feature = "cudnn")]
    add_cudnn_link_search();

    #[cfg(feature = "cuda")]
    {
        use std::path::PathBuf;
        let cuda_version_code = set_cuda_toolkit_version();
        if cuda_version_code.is_some_and(|version| version >= GDN_FP8_PRODUCER_MIN_CUDA) {
            println!("cargo:rustc-cfg=has_gdn_fp8_producer");
        }
        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rerun-if-changed=src/cuda");
        println!("cargo:rerun-if-env-changed=CUDA_NVCC_FLAGS");
        println!("cargo:rerun-if-env-changed=NVCC");
        println!("cargo:rerun-if-env-changed=CUDA_HOME");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
        let build_dir = cuda_build_dir(&out_dir, "kernels");
        let header_hash_arg = format!(
            "-DMISTRALRS_CORE_CUDA_HEADER_HASH=0x{:016x}",
            cuda_header_hash(std::path::Path::new("src/cuda"))
                .expect("failed to hash CUDA headers")
        );

        let mut builder = cudaforge::KernelBuilder::new()
            .source_glob("src/cuda/*.cu")
            .watch(["src/cuda"])
            .out_dir(&build_dir)
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
            .arg(&header_hash_arg);

        let compute_cap = builder.get_compute_cap().unwrap_or(80);

        // Check if CUDA_COMPUTE_CAP < 80 and disable bf16 kernels if so.
        // bf16 WMMA operations and certain bf16 intrinsics are only available on sm_80+.
        if compute_cap < 80 {
            builder = builder.arg("-DNO_BF16_KERNEL");
        }

        // https://github.com/EricLBuehler/mistral.rs/issues/286
        if let Some(cuda_nvcc_flags_env) = CUDA_NVCC_FLAGS {
            builder = builder.arg("--compiler-options");
            builder = builder.arg(cuda_nvcc_flags_env);
        }

        let target = std::env::var("TARGET").unwrap();

        // CUDA 13.x CCCL headers require MSVC's conforming preprocessor.
        if target.contains("msvc") {
            builder = builder.arg("--compiler-options").arg("/Zc:preprocessor");
        }

        // https://github.com/EricLBuehler/mistral.rs/issues/588
        let out_file = if target.contains("msvc") {
            // Windows case
            out_dir.join("mistralrscuda.lib")
        } else {
            out_dir.join("libmistralrscuda.a")
        };

        builder
            .build_lib(out_file)
            .expect("Build mistral-core failed!");
        println!("cargo:rustc-link-search={}", out_dir.display());
        println!("cargo:rustc-link-lib=mistralrscuda");
        println!("cargo:rustc-link-lib=dylib=cudart");

        if compute_cap == 90
            && target.contains("linux")
            && cuda_version_code.is_some_and(|version| version >= FLASHINFER_GDN_MIN_CUDA)
        {
            println!("cargo:rustc-cfg=has_flashinfer_gdn_sm90_kernel");
            println!("cargo:rerun-if-changed=third_party/flashinfer_gdn_sm90");
            let gdn_build_dir = cuda_build_dir(&out_dir, "flashinfer-gdn-sm90");
            let mut flashinfer_gdn = cudaforge::KernelBuilder::new()
                .source_files(["third_party/flashinfer_gdn_sm90/mistralrs_flashinfer_gdn_sm90.cu"])
                .out_dir(&gdn_build_dir)
                .compute_cap_arch("90a")
                .arg("-std=c++20")
                .arg("-O3")
                .arg("-U__CUDA_NO_HALF_OPERATORS__")
                .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
                .arg("-U__CUDA_NO_HALF2_OPERATORS__")
                .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
                .arg("--expt-relaxed-constexpr")
                .arg("--expt-extended-lambda")
                .arg("--use_fast_math")
                .arg("-static-global-template-stub=false")
                .arg("-Xfatbin=-compress-all")
                .arg("--compiler-options")
                .arg("-fPIC")
                .arg("-DFLAT_SM90A_ENABLED")
                .arg("-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED")
                .arg("-DFLASHINFER_ENABLE_BF16")
                .with_cutlass(Some(CUTLASS_COMMIT))
                .with_git_dependency(
                    "flashinfer-gdn",
                    "https://github.com/flashinfer-ai/flashinfer.git",
                    FLASHINFER_GDN_COMMIT,
                    vec!["include"],
                    false,
                );
            if let Some(cuda_nvcc_flags_env) = CUDA_NVCC_FLAGS {
                flashinfer_gdn = flashinfer_gdn
                    .arg("--compiler-options")
                    .arg(cuda_nvcc_flags_env);
            }
            flashinfer_gdn
                .build_lib(out_dir.join("libmistralrsflashinfergdn.a"))
                .expect("Build FlashInfer GDN provider failed!");
            println!("cargo:rustc-link-lib=mistralrsflashinfergdn");
        } else if compute_cap == 90
            && target.contains("linux")
            && cuda_version_code.is_none_or(|version| version < FLASHINFER_GDN_MIN_CUDA)
        {
            println!(
                "cargo:warning=FlashInfer GDN SM90 provider requires CUDA 12.8 or newer; using the native fallback"
            );
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
    }
}

#[cfg(feature = "cuda")]
fn set_cuda_toolkit_version() -> Option<u32> {
    let (version, code) = cuda_toolkit_version()?;
    {
        println!("cargo:rustc-env=MISTRALRS_BUILD_CUDA_VERSION={version}");
        println!("cargo:rustc-env=MISTRALRS_BUILD_CUDA_VERSION_CODE={code}");
    }
    Some(code)
}

#[cfg(feature = "cuda")]
fn cuda_toolkit_version() -> Option<(String, u32)> {
    let version = cudaforge::CudaToolkit::detect().ok()?.version?;
    parse_cuda_version(&version)
}

#[cfg(feature = "cuda")]
fn parse_cuda_version(version: &str) -> Option<(String, u32)> {
    let mut parts = version.split('.');
    let major: u32 = parts.next()?.parse().ok()?;
    let minor: u32 = parts.next().unwrap_or("0").parse().ok()?;
    Some((format!("{major}.{minor}"), major * 100 + minor))
}

#[cfg(feature = "cudnn")]
fn add_cudnn_link_search() {
    use std::path::PathBuf;

    println!("cargo:rerun-if-env-changed=CUDNN_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    let target = std::env::var("TARGET").unwrap_or_default();
    if !target.contains("msvc") {
        return;
    }

    if let Ok(dir) = std::env::var("CUDNN_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
        return;
    }

    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        candidates.push(PathBuf::from(&cuda_path).join("lib").join("x64"));
    }
    let cudnn_root = PathBuf::from(r"C:\Program Files\NVIDIA\CUDNN");
    if let Ok(versions) = std::fs::read_dir(&cudnn_root) {
        for version in versions.flatten() {
            let lib = version.path().join("lib");
            candidates.push(lib.join("x64"));
            if let Ok(cuda_vers) = std::fs::read_dir(&lib) {
                for cuda_ver in cuda_vers.flatten() {
                    candidates.push(cuda_ver.path().join("x64"));
                }
            }
        }
    }

    for dir in candidates {
        if dir.join("cudnn.lib").is_file() {
            println!("cargo:rustc-link-search=native={}", dir.display());
            return;
        }
    }

    println!(
        "cargo:warning=cudnn feature enabled but cudnn.lib not found; set CUDNN_LIB_DIR to its directory"
    );
}

fn set_git_revision() {
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=MISTRALRS_GIT_REVISION={commit}");
    println!("cargo:rerun-if-changed=.git/HEAD");
    if let Ok(head) = std::fs::read_to_string(".git/HEAD") {
        if let Some(ref_path) = head.strip_prefix("ref:") {
            let ref_path = ref_path.trim();
            if !ref_path.is_empty() {
                println!("cargo:rerun-if-changed=.git/{}", ref_path);
            }
        }
    }
}
