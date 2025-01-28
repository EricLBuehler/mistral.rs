#[cfg(feature = "cuda")]
const CUDA_NVCC_FLAGS: Option<&'static str> = option_env!("CUDA_NVCC_FLAGS");

const SUPPORTS_ATTN_SOFTMAX_FILE: &str = "src/utils/supports_attn_softmax.rs";

fn main() {
    #[cfg(feature = "cuda")]
    {
        use std::{path::PathBuf, vec};
        println!("cargo:rerun-if-changed=build.rs");
        let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
        let lib_files = vec!["src/cuda/nonzero_bitwise.cu"];
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
            .arg("--verbose");

        // https://github.com/EricLBuehler/mistral.rs/issues/286
        if let Some(cuda_nvcc_flags_env) = CUDA_NVCC_FLAGS {
            builder = builder.arg("--compiler-options");
            builder = builder.arg(cuda_nvcc_flags_env);
        }

        let target = std::env::var("TARGET").unwrap();

        // https://github.com/EricLBuehler/mistral.rs/issues/588
        let out_file = if target.contains("msvc") {
            // Windows case
            build_dir.join("mistralrscuda.lib")
        } else {
            build_dir.join("libmistralrscuda.a")
        };

        builder.build_lib(out_file);
        println!("cargo:rustc-link-search={}", build_dir.display());
        println!("cargo:rustc-link-lib=mistralrscuda");
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
    }

    #[cfg(feature = "metal")]
    {
        use std::fs::OpenOptions;
        use std::io::Write;
        use std::process::{Command, Stdio};

        // echo "__METAL_VERSION__" | xcrun -sdk macosx metal -E -x metal -P -

        // Create the `echo` command and pipe its output into `xcrun`
        let mut echo = Command::new("echo")
            .arg("__METAL_VERSION__")
            .stdout(Stdio::piped())
            .spawn()
            .expect("Failed to start echo command");

        echo.wait().unwrap();

        // Run the `xcrun` command, taking input from the `echo` command's output
        let output = Command::new("xcrun")
            .arg("-sdk")
            .arg("macosx")
            .arg("metal")
            .arg("-E")
            .arg("-x")
            .arg("metal")
            .arg("-P")
            .arg("-")
            .stdin(echo.stdout.unwrap())
            .output()
            .expect("Failed to run xcrun command");

        // Handle the output
        let supports_attn_softmax = if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout)
                .split('\n')
                .nth(1)
                .unwrap()
                .trim()
                .to_string()
                .parse::<usize>()
                .unwrap();
            // Attn softmax is only supported for metal >= 310 because of the vectorized bfloat types
            version >= 310
        } else {
            // Default to false if anything goes wrong
            false
        };

        let mut file = OpenOptions::new()
            .write(true)
            .open(SUPPORTS_ATTN_SOFTMAX_FILE)
            .unwrap();

        // Add the other stuff back
        if let Err(e) = writeln!(
            file,
            "pub(crate) const SUPPORTS_ATTN_SOFTMAX: bool = {supports_attn_softmax};"
        ) {
            panic!(
                "Error writing src/utils/supports_attn_softmax.rs: {:?}\n",
                e
            )
        }
    }

    #[cfg(not(feature = "metal"))]
    {
        use std::fs::OpenOptions;
        use std::io::Write;
        let mut file = OpenOptions::new()
            .write(true)
            .open(SUPPORTS_ATTN_SOFTMAX_FILE)
            .unwrap();

        // Add the other stuff back
        if let Err(e) = writeln!(
            file,
            "pub(crate) const SUPPORTS_ATTN_SOFTMAX: bool = false;"
        ) {
            panic!(
                "Error writing src/utils/supports_attn_softmax.rs: {:?}\n",
                e
            )
        }
    }
}
