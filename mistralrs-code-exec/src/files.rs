use std::{
    collections::{HashMap, HashSet},
    path::{Component, Path, PathBuf},
    time::SystemTime,
};

use base64::Engine;

use crate::protocol::{ExecuteFile, ExecuteOutputSpec};

const DEFAULT_MAX_OUTPUT_BYTES: u64 = 256 * 1024 * 1024;
const SKIPPED_COMPONENTS: &[&str] = &[
    "skills",
    "__pycache__",
    "node_modules",
    "target",
    "tmp",
    "temp",
    "cache",
    "venv",
    "env",
];
const SKIPPED_EXTENSIONS: &[&str] = &[
    "pyc", "pyo", "o", "so", "dylib", "dll", "rlib", "rmeta", "d", "class",
];

#[derive(Clone, Debug, Default)]
pub(crate) struct OutputSnapshot {
    files: HashMap<PathBuf, FileState>,
}

#[derive(Clone, Debug, PartialEq)]
struct FileState {
    len: u64,
    modified: Option<SystemTime>,
}

pub(crate) fn snapshot_output_files(work_dir: &Path) -> OutputSnapshot {
    OutputSnapshot {
        files: walk_output_files(work_dir).into_iter().collect(),
    }
}

pub(crate) fn collect_output_files(
    work_dir: &Path,
    outputs: &[ExecuteOutputSpec],
) -> Vec<ExecuteFile> {
    outputs
        .iter()
        .map(|spec| read_output_file(work_dir, spec))
        .collect()
}

pub(crate) fn append_auto_output_files(
    work_dir: &Path,
    snapshot: &OutputSnapshot,
    explicit_outputs: &[ExecuteOutputSpec],
    files: &mut Vec<ExecuteFile>,
) {
    let mut names = files
        .iter()
        .map(|file| file.name.clone())
        .chain(
            explicit_outputs
                .iter()
                .filter_map(|spec| normalized_output_name(&spec.name)),
        )
        .collect::<HashSet<_>>();

    for (path, state) in walk_output_files(work_dir) {
        if snapshot.files.get(&path) == Some(&state) {
            continue;
        }
        let Some(name) = path_to_output_name(&path) else {
            continue;
        };
        if !names.insert(name.clone()) {
            continue;
        }
        files.push(read_output_file(
            work_dir,
            &ExecuteOutputSpec { name, format: None },
        ));
    }
}

pub(crate) fn surface_outputs_json(work_dir: &Path, files: &[ExecuteFile]) -> String {
    let files_json: Vec<_> = files
        .iter()
        .map(|file| {
            let mut value = serde_json::json!({
                "name": &file.name,
                "format": &file.format,
                "bytes": file.size_bytes,
            });
            if let Some(error) = &file.error {
                value["status"] = serde_json::Value::String("error".to_string());
                value["error"] = serde_json::Value::String(error.clone());
            } else {
                value["status"] = serde_json::Value::String("success".to_string());
            }
            value
        })
        .collect();
    let status = if files.iter().any(ExecuteFile::is_error) {
        "error"
    } else {
        "success"
    };
    serde_json::json!({
        "status": status,
        "working_directory": work_dir.display().to_string(),
        "files": files_json,
    })
    .to_string()
}

pub(crate) fn read_output_file(work_dir: &Path, spec: &ExecuteOutputSpec) -> ExecuteFile {
    let name = spec.name.clone();
    let format = output_format(spec);
    let mut mime_type = Some(mime_for_format(&format).to_string());
    let mut out = ExecuteFile {
        name: name.clone(),
        format,
        mime_type: mime_type.clone(),
        size_bytes: 0,
        text: None,
        data_base64: None,
        error: None,
    };

    if name.is_empty() {
        out.error = Some("missing name".to_string());
        return out;
    }

    let path = match output_path(work_dir, &name) {
        Ok(path) => path,
        Err(e) => {
            out.error = Some(e.to_string());
            return out;
        }
    };

    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            out.error = Some("not produced".to_string());
            return out;
        }
        Err(e) => {
            out.error = Some(format!("read failed: {e}"));
            return out;
        }
    };

    out.size_bytes = bytes.len() as u64;
    let max_output_bytes = max_output_bytes();
    if out.size_bytes > max_output_bytes {
        out.error = Some(format!(
            "exceeds max output size ({} bytes; cap is {} bytes)",
            out.size_bytes, max_output_bytes
        ));
        return out;
    }

    if is_text_format(&out.format) {
        match String::from_utf8(bytes) {
            Ok(text) => out.text = Some(text),
            Err(e) => {
                out.data_base64 =
                    Some(base64::engine::general_purpose::STANDARD.encode(e.into_bytes()));
                mime_type = Some("application/octet-stream".to_string());
            }
        }
    } else {
        out.data_base64 = Some(base64::engine::general_purpose::STANDARD.encode(bytes));
    }
    out.mime_type = mime_type;
    out
}

fn walk_output_files(work_dir: &Path) -> Vec<(PathBuf, FileState)> {
    let mut files = Vec::new();
    visit_output_dir(work_dir, work_dir, &mut files);
    files.sort_by(|a, b| a.0.cmp(&b.0));
    files
}

fn visit_output_dir(work_dir: &Path, dir: &Path, files: &mut Vec<(PathBuf, FileState)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(rel) = path.strip_prefix(work_dir) else {
            continue;
        };
        if should_skip_auto_path(rel) {
            continue;
        }
        let Ok(meta) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        let file_type = meta.file_type();
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_dir() {
            visit_output_dir(work_dir, &path, files);
        } else if file_type.is_file() {
            files.push((
                rel.to_path_buf(),
                FileState {
                    len: meta.len(),
                    modified: meta.modified().ok(),
                },
            ));
        }
    }
}

fn should_skip_auto_path(path: &Path) -> bool {
    let mut last = None;
    for component in path.components() {
        let Component::Normal(name) = component else {
            return true;
        };
        let Some(name) = name.to_str() else {
            return true;
        };
        if name.is_empty() || name.starts_with('.') || SKIPPED_COMPONENTS.contains(&name) {
            return true;
        }
        last = Some(name);
    }
    last.and_then(|name| name.rsplit_once('.').map(|(_, ext)| ext))
        .is_some_and(|ext| SKIPPED_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()))
}

fn output_path(work_dir: &Path, name: &str) -> anyhow::Result<PathBuf> {
    let path = Path::new(name);
    if path.is_absolute()
        || path.components().any(|c| {
            matches!(
                c,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        anyhow::bail!("output path must be relative to the working directory");
    }
    Ok(work_dir.join(path))
}

fn normalized_output_name(name: &str) -> Option<String> {
    if name.is_empty() {
        return None;
    }
    let path = Path::new(name);
    if path.is_absolute() {
        return None;
    }
    path_to_output_name(path)
}

fn path_to_output_name(path: &Path) -> Option<String> {
    let mut parts = Vec::new();
    for component in path.components() {
        let Component::Normal(name) = component else {
            return None;
        };
        parts.push(name.to_str()?);
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("/"))
    }
}

fn output_format(spec: &ExecuteOutputSpec) -> String {
    spec.format
        .clone()
        .or_else(|| {
            Path::new(&spec.name)
                .extension()
                .and_then(|ext| ext.to_str())
                .map(ToString::to_string)
        })
        .unwrap_or_default()
        .to_ascii_lowercase()
}

fn max_output_bytes() -> u64 {
    std::env::var("MISTRALRS_MAX_OUTPUT_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_OUTPUT_BYTES)
}

fn is_text_format(format: &str) -> bool {
    matches!(
        format,
        "csv"
            | "tsv"
            | "json"
            | "geojson"
            | "xml"
            | "yaml"
            | "yml"
            | "toml"
            | "md"
            | "markdown"
            | "html"
            | "htm"
            | "svg"
            | "latex"
            | "tex"
            | "sql"
            | "python"
            | "py"
            | "rust"
            | "rs"
            | "txt"
            | "text"
            | "log"
            | "vega"
            | "vega-lite"
    )
}

fn mime_for_format(format: &str) -> &'static str {
    match format {
        "csv" => "text/csv",
        "tsv" => "text/tab-separated-values",
        "json" | "geojson" | "vega" | "vega-lite" => "application/json",
        "xml" => "application/xml",
        "yaml" | "yml" => "application/yaml",
        "toml" => "application/toml",
        "md" | "markdown" => "text/markdown",
        "html" | "htm" => "text/html",
        "svg" => "image/svg+xml",
        "latex" | "tex" => "application/x-tex",
        "sql" => "application/sql",
        "python" | "py" => "text/x-python",
        "rust" | "rs" => "text/x-rust",
        "txt" | "text" | "log" => "text/plain",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "pdf" => "application/pdf",
        "pptx" => "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xlsx" => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "parquet" => "application/x-parquet",
        "zip" => "application/zip",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_outputs_include_nested_new_files() {
        let dir = tempfile::tempdir().unwrap();
        let snapshot = snapshot_output_files(dir.path());

        std::fs::create_dir_all(dir.path().join("outputs/assets")).unwrap();
        std::fs::write(dir.path().join("outputs/assets/chart.svg"), "<svg></svg>").unwrap();

        let mut files = Vec::new();
        append_auto_output_files(dir.path(), &snapshot, &[], &mut files);

        assert_eq!(files.len(), 1);
        assert_eq!(files[0].name, "outputs/assets/chart.svg");
        assert_eq!(files[0].format, "svg");
        assert_eq!(files[0].mime_type.as_deref(), Some("image/svg+xml"));
        assert_eq!(files[0].text.as_deref(), Some("<svg></svg>"));
    }

    #[test]
    fn auto_outputs_include_modified_existing_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("report.txt");
        std::fs::write(&path, "before").unwrap();
        let snapshot = snapshot_output_files(dir.path());

        std::fs::write(&path, "after").unwrap();

        let mut files = Vec::new();
        append_auto_output_files(dir.path(), &snapshot, &[], &mut files);

        assert_eq!(files.len(), 1);
        assert_eq!(files[0].name, "report.txt");
        assert_eq!(files[0].text.as_deref(), Some("after"));
    }

    #[test]
    fn auto_outputs_skip_dependency_and_cache_paths() {
        let dir = tempfile::tempdir().unwrap();
        let snapshot = snapshot_output_files(dir.path());

        for path in [
            "skills/example/SKILL.md",
            "__pycache__/module.pyc",
            ".hidden/output.txt",
            "tmp/scratch.txt",
            "node_modules/package/index.js",
        ] {
            let path = dir.path().join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, "junk").unwrap();
        }
        std::fs::write(dir.path().join("keep.txt"), "ok").unwrap();

        let mut files = Vec::new();
        append_auto_output_files(dir.path(), &snapshot, &[], &mut files);

        assert_eq!(files.len(), 1);
        assert_eq!(files[0].name, "keep.txt");
    }

    #[test]
    fn auto_outputs_do_not_duplicate_explicit_outputs() {
        let dir = tempfile::tempdir().unwrap();
        let snapshot = snapshot_output_files(dir.path());
        std::fs::write(dir.path().join("answer.txt"), "ok").unwrap();

        let outputs = vec![ExecuteOutputSpec {
            name: "answer.txt".to_string(),
            format: None,
        }];
        let mut files = collect_output_files(dir.path(), &outputs);
        append_auto_output_files(dir.path(), &snapshot, &outputs, &mut files);

        assert_eq!(files.len(), 1);
        assert_eq!(files[0].name, "answer.txt");
    }
}
