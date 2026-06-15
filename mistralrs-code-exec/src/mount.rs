use std::{
    collections::HashSet,
    ffi::OsStr,
    path::{Path, PathBuf},
};

use anyhow::Context;
use base64::{engine::general_purpose::STANDARD, Engine};
use mistralrs_mcp::ToolInputFile;

pub fn mount_input_files(work_dir: &Path, files: &[ToolInputFile]) -> anyhow::Result<()> {
    if files.is_empty() {
        return Ok(());
    }
    std::fs::create_dir_all(work_dir)?;
    let mut used = HashSet::new();
    for file in files {
        let name = unique_name(&safe_file_name(&file.name), &mut used);
        let dest = work_dir.join(&name);
        if let Some(text) = &file.text {
            std::fs::write(&dest, text).with_context(|| format!("write input file {name}"))?;
        } else if let Some(data_base64) = &file.data_base64 {
            let bytes = STANDARD
                .decode(data_base64)
                .with_context(|| format!("decode input file {}", file.id))?;
            std::fs::write(&dest, bytes).with_context(|| format!("write input file {name}"))?;
        }
    }
    Ok(())
}

fn safe_file_name(name: &str) -> String {
    let base = Path::new(name)
        .file_name()
        .and_then(OsStr::to_str)
        .unwrap_or("file");
    let cleaned: String = base
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.is_empty() || cleaned == "." || cleaned == ".." {
        "file".to_string()
    } else {
        cleaned
    }
}

fn unique_name(name: &str, used: &mut HashSet<String>) -> String {
    if used.insert(name.to_string()) {
        return name.to_string();
    }
    let path = PathBuf::from(name);
    let stem = path.file_stem().and_then(OsStr::to_str).unwrap_or("file");
    let ext = path.extension().and_then(OsStr::to_str);
    for idx in 2usize.. {
        let candidate = match ext {
            Some(ext) if !ext.is_empty() => format!("{stem}-{idx}.{ext}"),
            _ => format!("{stem}-{idx}"),
        };
        if used.insert(candidate.clone()) {
            return candidate;
        }
    }
    unreachable!()
}
