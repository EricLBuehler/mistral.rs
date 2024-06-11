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
        let builder = bindgen_cuda::Builder::default()
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
        let out_file = build_dir.join("libmistralcuda.a");
        builder.build_lib(out_file);
        println!("cargo:rustc-link-search={}", build_dir.display());
        println!("cargo:rustc-link-lib=mistralcuda");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}
