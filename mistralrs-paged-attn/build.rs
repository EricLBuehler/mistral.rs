use anyhow::Result;

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn main() -> Result<()> {
    use std::fs;
    use std::fs::read_to_string;
    use std::fs::OpenOptions;
    use std::io::prelude::*;
    use std::path::PathBuf;

    const OTHER_CONTENT: &str = r#"
#[cfg(all(feature = "cuda", target_family = "unix"))]
mod ffi;
#[cfg(all(feature = "cuda", target_family = "unix"))]
mod backend;

#[cfg(all(feature = "cuda", target_family = "unix"))]
pub use backend::{{copy_blocks, paged_attention, reshape_and_cache, swap_blocks}};
    "#;

    fn read_lines(filename: &str) -> Vec<String> {
        let mut result = Vec::new();

        for line in read_to_string(filename).unwrap().lines() {
            result.push(line.to_string())
        }

        result
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/pagedattention.cu");
    println!("cargo:rerun-if-changed=src/copy_blocks_kernel.cu");
    println!("cargo:rerun-if-changed=src/reshape_and_cache_kernel.cu");
    let builder = bindgen_cuda::Builder::default();
    println!("cargo:info={builder:?}");
    builder.build_lib("libpagedattention.a");

    let bindings = builder.build_ptx().unwrap();
    bindings.write("src/lib.rs").unwrap();

    let kernel_dir = PathBuf::from("../mistralrs-paged-attn");
    let absolute_kernel_dir = std::fs::canonicalize(kernel_dir).unwrap();

    println!(
        "cargo:rustc-link-search=native={}",
        absolute_kernel_dir.display()
    );
    println!("cargo:rustc-link-lib=pagedattention");
    println!("cargo:rustc-link-lib=dylib=cudart");

    let contents = read_lines("src/lib.rs");
    for line in contents {
        if line == "pub mod ffi;" {
            return Ok(());
        }
    }
    let ct = fs::read_to_string("src/lib.rs")?;
    if !ct.contains(OTHER_CONTENT) {
        let mut file = OpenOptions::new().append(true).open("src/lib.rs").unwrap();

        // Add the other stuff back
        if let Err(e) = writeln!(file, "{OTHER_CONTENT}") {
            anyhow::bail!("Error while building dependencies: {:?}\n", e)
        }
    }
    Ok(())
}

#[cfg(not(all(feature = "cuda", target_family = "unix")))]
fn main() -> Result<()> {
    Ok(())
}
