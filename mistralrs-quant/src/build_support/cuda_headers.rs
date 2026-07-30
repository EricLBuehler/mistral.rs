use std::{
    fs, io,
    path::{Path, PathBuf},
};

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

pub fn find(root: &Path) -> io::Result<Vec<PathBuf>> {
    fn visit(path: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
        if path.is_dir() {
            for entry in fs::read_dir(path)? {
                visit(&entry?.path(), files)?;
            }
        } else if matches!(
            path.extension().and_then(|extension| extension.to_str()),
            Some("h" | "cuh" | "hpp")
        ) {
            files.push(path.to_path_buf());
        }
        Ok(())
    }

    let mut files = Vec::new();
    visit(root, &mut files)?;
    files.sort();
    Ok(files)
}

pub fn hash(files: &[PathBuf]) -> io::Result<u64> {
    fn update(hash: &mut u64, bytes: &[u8]) {
        for &byte in bytes {
            *hash ^= u64::from(byte);
            *hash = hash.wrapping_mul(FNV_PRIME);
        }
    }

    let mut hash = FNV_OFFSET_BASIS;
    for file in files {
        let path = file.to_string_lossy();
        let contents = fs::read(file)?;
        update(&mut hash, &(path.len() as u64).to_le_bytes());
        update(&mut hash, path.as_bytes());
        update(&mut hash, &(contents.len() as u64).to_le_bytes());
        update(&mut hash, &contents);
    }
    Ok(hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_supported_extensions_in_sorted_order() -> io::Result<()> {
        let dir = tempfile::tempdir()?;
        fs::create_dir(dir.path().join("nested"))?;
        for file in ["z.hpp", "a.h", "nested/c.cuh", "ignored.cu", "ignored.txt"] {
            fs::write(dir.path().join(file), file)?;
        }

        let files = find(dir.path())?;
        let relative = files
            .iter()
            .map(|file| file.strip_prefix(dir.path()).unwrap().to_path_buf())
            .collect::<Vec<_>>();
        assert_eq!(
            relative,
            [
                PathBuf::from("a.h"),
                PathBuf::from("nested").join("c.cuh"),
                PathBuf::from("z.hpp"),
            ]
        );
        Ok(())
    }

    #[test]
    fn content_change_changes_hash() -> io::Result<()> {
        let dir = tempfile::tempdir()?;
        let header = dir.path().join("kernel.cuh");
        fs::write(&header, "before")?;
        let files = find(dir.path())?;
        let before = hash(&files)?;

        fs::write(&header, "after")?;
        assert_ne!(before, hash(&files)?);
        Ok(())
    }

    #[test]
    fn path_change_changes_hash() -> io::Result<()> {
        let dir = tempfile::tempdir()?;
        let first = dir.path().join("first.h");
        fs::write(&first, "same")?;
        let before = hash(&find(dir.path())?)?;

        fs::rename(first, dir.path().join("second.h"))?;
        assert_ne!(before, hash(&find(dir.path())?)?);
        Ok(())
    }
}
