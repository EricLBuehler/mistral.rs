use std::{
    fs, io,
    path::{Path, PathBuf},
};

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

fn collect_files(root: &Path) -> io::Result<Vec<PathBuf>> {
    fn visit(path: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
        if path.is_dir() {
            for entry in fs::read_dir(path)? {
                visit(&entry?.path(), files)?;
            }
        } else if path.is_file() {
            files.push(path.to_path_buf());
        }
        Ok(())
    }

    let mut files = Vec::new();
    visit(root, &mut files)?;
    files.sort();
    Ok(files)
}

fn update_hash(hash: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

fn push_entry(
    output: &mut String,
    source_hash: &mut u64,
    installed_path: &Path,
    source_path: &Path,
) -> io::Result<()> {
    let installed_path = installed_path.to_string_lossy();
    let contents = fs::read(source_path)?;
    update_hash(source_hash, &(installed_path.len() as u64).to_le_bytes());
    update_hash(source_hash, installed_path.as_bytes());
    update_hash(source_hash, &(contents.len() as u64).to_le_bytes());
    update_hash(source_hash, &contents);

    let source_path = source_path.canonicalize()?;
    output.push_str("    (");
    output.push_str(&format!("{installed_path:?}"));
    output.push_str(", include_bytes!(");
    output.push_str(&format!("{:?}", source_path.to_string_lossy()));
    output.push_str(")),\n");
    Ok(())
}

pub fn write(
    output_path: &Path,
    official_include_root: &Path,
    skinny_include_root: &Path,
    cutlass_include_root: &Path,
    generator_hash: u64,
) -> io::Result<u64> {
    let mut output = String::from("const DEEPGEMM_JIT_HEADERS: &[(&str, &[u8])] = &[\n");
    let mut source_hash = FNV_OFFSET_BASIS;
    update_hash(&mut source_hash, &generator_hash.to_le_bytes());
    for source in collect_files(official_include_root)? {
        let relative = source.strip_prefix(official_include_root).unwrap();
        push_entry(
            &mut output,
            &mut source_hash,
            &Path::new("official").join(relative),
            &source,
        )?;
    }
    for source in collect_files(skinny_include_root)? {
        let relative = source.strip_prefix(skinny_include_root).unwrap();
        push_entry(
            &mut output,
            &mut source_hash,
            &Path::new("skinny/deep_gemm").join(relative),
            &source,
        )?;
    }
    for source in collect_files(cutlass_include_root)? {
        let relative = source.strip_prefix(cutlass_include_root).unwrap();
        push_entry(
            &mut output,
            &mut source_hash,
            &Path::new("official").join(relative),
            &source,
        )?;
    }
    output.push_str("];\n");
    fs::write(output_path, output)?;
    Ok(source_hash)
}
