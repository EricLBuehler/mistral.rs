use std::{
    collections::HashMap,
    fs,
    io::{Cursor, Read},
    path::{Component, Path, PathBuf},
    sync::{Arc, RwLock},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{bail, Context, Result};
use axum::{
    extract::{Multipart, Path as AxumPath, Query},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    Extension, Json,
};
use chrono::{DateTime, SecondsFormat, Utc};
use mistralrs_core::ShellSkillMount;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use utoipa::ToSchema;

use crate::openai::OpenAiShellSkillReference;

const SKILL_OBJECT: &str = "skill";
const SKILL_VERSION_OBJECT: &str = "skill.version";
const ANTHROPIC_SKILL_VERSION_OBJECT: &str = "skill_version";
const CUSTOM_SKILL_SOURCE: &str = "custom";
const ANTHROPIC_SKILL_SOURCE: &str = "anthropic";
const SKILL_METADATA_FILE: &str = "skill.json";
const SKILL_CONTENT_DIR: &str = "content";
const MAX_SKILL_UPLOAD_BYTES: usize = 50 * 1024 * 1024;
const MAX_SKILL_FILES: usize = 500;

#[derive(Clone)]
pub struct SkillStore {
    root: PathBuf,
    skills: Arc<RwLock<HashMap<String, SkillMetadata>>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SkillMetadata {
    id: String,
    name: String,
    description: String,
    created_at: u64,
    versions: Vec<SkillVersionMetadata>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SkillVersionMetadata {
    version: u64,
    created_at: u64,
    source_path: PathBuf,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct SkillObject {
    pub id: String,
    pub object: &'static str,
    pub created_at: u64,
    pub name: String,
    pub description: String,
    pub latest_version: u64,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct SkillVersionObject {
    pub id: String,
    pub object: &'static str,
    pub skill_id: String,
    pub created_at: u64,
    pub version: u64,
    pub name: String,
    pub description: String,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct SkillListObject {
    pub object: &'static str,
    pub data: Vec<SkillObject>,
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
pub struct SkillListQuery {
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub page: Option<String>,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct AnthropicSkillObject {
    pub id: String,
    #[serde(rename = "type")]
    pub tp: &'static str,
    pub created_at: String,
    pub updated_at: String,
    pub display_title: String,
    pub latest_version: String,
    pub source: &'static str,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct AnthropicSkillListObject {
    pub data: Vec<AnthropicSkillObject>,
    pub has_more: bool,
    pub next_page: Option<String>,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct AnthropicSkillVersionObject {
    pub id: String,
    #[serde(rename = "type")]
    pub tp: &'static str,
    pub skill_id: String,
    pub created_at: String,
    pub version: String,
    pub name: String,
    pub description: String,
    pub directory: String,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct AnthropicSkillVersionListObject {
    pub data: Vec<AnthropicSkillVersionObject>,
    pub has_more: bool,
    pub next_page: Option<String>,
}

struct SkillUpload {
    name: String,
    description: String,
    source_path: PathBuf,
    staging_path: PathBuf,
}

impl SkillStore {
    pub fn default_root() -> PathBuf {
        std::env::temp_dir().join("mistralrs-skills")
    }

    pub fn new(root: PathBuf) -> Result<Self> {
        fs::create_dir_all(&root)?;
        let store = Self {
            root,
            skills: Arc::new(RwLock::new(HashMap::new())),
        };
        store.load()?;
        Ok(store)
    }

    pub fn list(&self) -> Result<Vec<SkillObject>> {
        let skills = self
            .skills
            .read()
            .map_err(|_| anyhow::anyhow!("skill store lock poisoned"))?;
        Ok(skills.values().map(SkillObject::from).collect())
    }

    pub fn list_versions(&self, skill_id: &str) -> Result<Vec<SkillVersionObject>> {
        let skills = self
            .skills
            .read()
            .map_err(|_| anyhow::anyhow!("skill store lock poisoned"))?;
        let metadata = skills
            .get(skill_id)
            .ok_or_else(|| anyhow::anyhow!("Skill `{skill_id}` was not found."))?;
        metadata
            .versions
            .iter()
            .map(|version| SkillVersionObject::from_metadata(metadata, version.version))
            .collect()
    }

    pub async fn create_skill(&self, multipart: Multipart) -> Result<SkillObject> {
        let upload = parse_upload(multipart).await?;
        let id = format!("skill_{}", uuid::Uuid::new_v4().simple());
        let created_at = unix_now();
        let source_path = self.store_version_content(&id, 1, &upload.source_path)?;
        let _ = fs::remove_dir_all(&upload.staging_path);
        let metadata = SkillMetadata {
            id: id.clone(),
            name: upload.name,
            description: upload.description,
            created_at,
            versions: vec![SkillVersionMetadata {
                version: 1,
                created_at,
                source_path,
            }],
        };
        self.persist_metadata(&metadata)?;
        self.skills
            .write()
            .map_err(|_| anyhow::anyhow!("skill store lock poisoned"))?
            .insert(id, metadata.clone());
        Ok(SkillObject::from(&metadata))
    }

    pub async fn create_version(
        &self,
        skill_id: &str,
        multipart: Multipart,
    ) -> Result<SkillVersionObject> {
        let upload = parse_upload(multipart).await?;
        let mut skills = self
            .skills
            .write()
            .map_err(|_| anyhow::anyhow!("skill store lock poisoned"))?;
        let metadata = skills
            .get_mut(skill_id)
            .ok_or_else(|| anyhow::anyhow!("Skill `{skill_id}` was not found."))?;
        let version = metadata.versions.last().map(|v| v.version + 1).unwrap_or(1);
        let created_at = unix_now();
        let source_path = self.store_version_content(skill_id, version, &upload.source_path)?;
        let _ = fs::remove_dir_all(&upload.staging_path);
        metadata.name = upload.name;
        metadata.description = upload.description;
        metadata.versions.push(SkillVersionMetadata {
            version,
            created_at,
            source_path,
        });
        self.persist_metadata(metadata)?;
        SkillVersionObject::from_metadata(metadata, version)
    }

    pub fn resolve_references(
        &self,
        refs: &[OpenAiShellSkillReference],
    ) -> Result<mistralrs_core::ShellOptions> {
        let mut skills = Vec::new();
        for reference in refs {
            skills.push(self.resolve_reference(reference)?);
        }
        Ok(mistralrs_core::ShellOptions { skills })
    }

    fn resolve_reference(&self, reference: &OpenAiShellSkillReference) -> Result<ShellSkillMount> {
        let skills = self
            .skills
            .read()
            .map_err(|_| anyhow::anyhow!("skill store lock poisoned"))?;
        let metadata = skills
            .get(&reference.skill_id)
            .ok_or_else(|| anyhow::anyhow!("Skill `{}` was not found.", reference.skill_id))?;
        let version = match &reference.version {
            None => metadata.versions.last(),
            Some(Value::String(s)) if s == "latest" => metadata.versions.last(),
            Some(Value::String(s)) => {
                let parsed = s
                    .parse::<u64>()
                    .with_context(|| format!("Invalid skill version `{s}`"))?;
                metadata
                    .versions
                    .iter()
                    .find(|version| version.version == parsed)
            }
            Some(Value::Number(n)) => {
                let parsed = n
                    .as_u64()
                    .ok_or_else(|| anyhow::anyhow!("Invalid skill version `{n}`"))?;
                metadata
                    .versions
                    .iter()
                    .find(|version| version.version == parsed)
            }
            Some(other) => bail!("Unsupported skill version value `{other}`"),
        }
        .ok_or_else(|| anyhow::anyhow!("Skill `{}` version was not found.", reference.skill_id))?;

        Ok(ShellSkillMount {
            name: metadata.name.clone(),
            description: metadata.description.clone(),
            source_path: version.source_path.clone(),
        })
    }

    fn store_version_content(
        &self,
        skill_id: &str,
        version: u64,
        source: &Path,
    ) -> Result<PathBuf> {
        let version_root = self
            .root
            .join(skill_id)
            .join("versions")
            .join(version.to_string());
        let content_dir = version_root.join(SKILL_CONTENT_DIR);
        if content_dir.exists() {
            fs::remove_dir_all(&content_dir)?;
        }
        copy_dir_all(source, &content_dir)?;
        Ok(content_dir)
    }

    fn persist_metadata(&self, metadata: &SkillMetadata) -> Result<()> {
        let skill_dir = self.root.join(&metadata.id);
        fs::create_dir_all(&skill_dir)?;
        let bytes = serde_json::to_vec_pretty(metadata)?;
        fs::write(skill_dir.join(SKILL_METADATA_FILE), bytes)?;
        Ok(())
    }

    fn load(&self) -> Result<()> {
        let mut loaded = HashMap::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let metadata_path = entry.path().join(SKILL_METADATA_FILE);
            if !metadata_path.exists() {
                continue;
            }
            let bytes = fs::read(&metadata_path)?;
            let metadata: SkillMetadata = serde_json::from_slice(&bytes)
                .with_context(|| format!("parse {}", metadata_path.display()))?;
            loaded.insert(metadata.id.clone(), metadata);
        }
        *self
            .skills
            .write()
            .map_err(|_| anyhow::anyhow!("skill store lock poisoned"))? = loaded;
        Ok(())
    }
}

impl From<&SkillMetadata> for SkillObject {
    fn from(value: &SkillMetadata) -> Self {
        Self {
            id: value.id.clone(),
            object: SKILL_OBJECT,
            created_at: value.created_at,
            name: value.name.clone(),
            description: value.description.clone(),
            latest_version: value.versions.last().map(|v| v.version).unwrap_or(0),
        }
    }
}

impl From<&SkillObject> for AnthropicSkillObject {
    fn from(value: &SkillObject) -> Self {
        let created_at = unix_to_rfc3339(value.created_at);
        Self {
            id: value.id.clone(),
            tp: SKILL_OBJECT,
            created_at: created_at.clone(),
            updated_at: created_at,
            display_title: value.name.clone(),
            latest_version: value.latest_version.to_string(),
            source: CUSTOM_SKILL_SOURCE,
        }
    }
}

impl SkillVersionObject {
    fn from_metadata(metadata: &SkillMetadata, version: u64) -> Result<Self> {
        let version_metadata = metadata
            .versions
            .iter()
            .find(|v| v.version == version)
            .ok_or_else(|| {
                anyhow::anyhow!("Skill `{}` version `{version}` missing", metadata.id)
            })?;
        Ok(Self {
            id: format!("{}_v{}", metadata.id, version),
            object: SKILL_VERSION_OBJECT,
            skill_id: metadata.id.clone(),
            created_at: version_metadata.created_at,
            version,
            name: metadata.name.clone(),
            description: metadata.description.clone(),
        })
    }
}

impl From<&SkillVersionObject> for AnthropicSkillVersionObject {
    fn from(value: &SkillVersionObject) -> Self {
        Self {
            id: value.id.clone(),
            tp: ANTHROPIC_SKILL_VERSION_OBJECT,
            skill_id: value.skill_id.clone(),
            created_at: unix_to_rfc3339(value.created_at),
            version: value.version.to_string(),
            name: value.name.clone(),
            description: value.description.clone(),
            directory: value.name.clone(),
        }
    }
}

async fn parse_upload(mut multipart: Multipart) -> Result<SkillUpload> {
    let staging = tempfile::Builder::new()
        .prefix("mistralrs-skill-upload-")
        .tempdir()?;
    let mut files = Vec::new();
    let mut total_bytes = 0usize;

    while let Some(field) = multipart.next_field().await? {
        let field_name = field.name().unwrap_or_default().to_string();
        if field_name != "files" && field_name != "file" && field_name != "files[]" {
            continue;
        }
        let file_name = field
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("Uploaded skill file is missing a filename."))?
            .to_string();
        let bytes = field.bytes().await?;
        total_bytes = total_bytes.saturating_add(bytes.len());
        if total_bytes > MAX_SKILL_UPLOAD_BYTES {
            bail!("Skill upload exceeds the {MAX_SKILL_UPLOAD_BYTES} byte limit.");
        }
        files.push((file_name, bytes.to_vec()));
        if files.len() > MAX_SKILL_FILES {
            bail!("Skill upload may contain at most {MAX_SKILL_FILES} files.");
        }
    }

    if files.is_empty() {
        bail!("Skill upload requires multipart file field `files`.");
    }

    if files.len() == 1 && looks_like_zip(&files[0].0, &files[0].1) {
        extract_zip(&files[0].1, staging.path())?;
    } else {
        for (file_name, bytes) in files {
            let rel = safe_relative_path(&file_name)?;
            let dest = staging.path().join(rel);
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(dest, bytes)?;
        }
    }

    let skill_root = find_skill_root(staging.path())?;
    let (name, description) = read_skill_metadata(&skill_root.join("SKILL.md"))?;
    let skill_root_rel = skill_root
        .strip_prefix(staging.path())
        .unwrap_or_else(|_| Path::new(""))
        .to_path_buf();
    let persisted_staging = staging.keep();
    Ok(SkillUpload {
        name,
        description,
        source_path: persisted_staging.join(skill_root_rel),
        staging_path: persisted_staging,
    })
}

fn looks_like_zip(file_name: &str, bytes: &[u8]) -> bool {
    file_name.ends_with(".zip") || bytes.starts_with(b"PK\x03\x04")
}

fn extract_zip(bytes: &[u8], dest: &Path) -> Result<()> {
    let reader = Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(reader)?;
    if archive.len() > MAX_SKILL_FILES {
        bail!("Skill zip may contain at most {MAX_SKILL_FILES} files.");
    }
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        if file.is_dir() {
            continue;
        }
        if file
            .unix_mode()
            .is_some_and(|mode| mode & 0o170000 == 0o120000)
        {
            bail!("Skill zip may not contain symlinks.");
        }
        let rel = file
            .enclosed_name()
            .ok_or_else(|| anyhow::anyhow!("Skill zip contains an unsafe path."))?;
        let out = dest.join(rel);
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        fs::write(out, buf)?;
    }
    Ok(())
}

fn safe_relative_path(path: &str) -> Result<PathBuf> {
    let path = Path::new(path);
    if path.is_absolute() {
        bail!("Skill file paths must be relative.");
    }
    let mut clean = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => clean.push(part),
            Component::CurDir => {}
            _ => bail!("Skill file path `{}` is not allowed.", path.display()),
        }
    }
    if clean.as_os_str().is_empty() {
        bail!("Skill file path may not be empty.");
    }
    Ok(clean)
}

fn find_skill_root(staging: &Path) -> Result<PathBuf> {
    if staging.join("SKILL.md").is_file() {
        return Ok(staging.to_path_buf());
    }
    let entries = fs::read_dir(staging)?
        .filter_map(|entry| entry.ok())
        .collect::<Vec<_>>();
    let dirs = entries
        .iter()
        .filter(|entry| entry.file_type().is_ok_and(|tp| tp.is_dir()))
        .collect::<Vec<_>>();
    if dirs.len() != 1 {
        bail!("Skill upload must contain exactly one top-level folder with SKILL.md.");
    }
    let root = dirs[0].path();
    if !root.join("SKILL.md").is_file() {
        bail!("Skill upload top-level folder must contain SKILL.md.");
    }
    Ok(root)
}

fn read_skill_metadata(skill_md: &Path) -> Result<(String, String)> {
    let text =
        fs::read_to_string(skill_md).with_context(|| format!("read {}", skill_md.display()))?;
    let Some(frontmatter) = text
        .strip_prefix("---")
        .and_then(|rest| rest.split_once("---").map(|(meta, _)| meta))
    else {
        bail!("SKILL.md must start with YAML frontmatter containing `name` and `description`.");
    };
    let mut name = None;
    let mut description = None;
    for line in frontmatter.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .to_string();
        match key.trim() {
            "name" => name = Some(value),
            "description" => description = Some(value),
            _ => {}
        }
    }
    let name = name.ok_or_else(|| anyhow::anyhow!("SKILL.md frontmatter is missing `name`."))?;
    let description = description
        .ok_or_else(|| anyhow::anyhow!("SKILL.md frontmatter is missing `description`."))?;
    if name.trim().is_empty() || description.trim().is_empty() {
        bail!("SKILL.md `name` and `description` must be non-empty.");
    }
    Ok((name, description))
}

fn copy_dir_all(src: &Path, dst: &Path) -> Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let dest = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_all(&entry.path(), &dest)?;
        } else if file_type.is_file() {
            fs::copy(entry.path(), dest)?;
        }
    }
    Ok(())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn unix_to_rfc3339(timestamp: u64) -> String {
    DateTime::<Utc>::from_timestamp(timestamp as i64, 0)
        .unwrap_or(DateTime::<Utc>::UNIX_EPOCH)
        .to_rfc3339_opts(SecondsFormat::Secs, true)
}

fn skill_error(status: StatusCode, error: anyhow::Error) -> axum::response::Response {
    (
        status,
        Json(json!({
            "error": {
                "message": error.to_string(),
                "type": "invalid_request_error"
            }
        })),
    )
        .into_response()
}

fn prefers_anthropic_shape(headers: &HeaderMap, query: Option<&SkillListQuery>) -> bool {
    headers.contains_key("anthropic-version")
        || headers.contains_key("anthropic-beta")
        || query.and_then(|query| query.source.as_deref()).is_some()
}

fn anthropic_list_response(
    skills: Vec<SkillObject>,
    query: Option<&SkillListQuery>,
) -> AnthropicSkillListObject {
    let mut data = match query.and_then(|query| query.source.as_deref()) {
        Some(ANTHROPIC_SKILL_SOURCE) => Vec::new(),
        Some(CUSTOM_SKILL_SOURCE) | None => skills
            .iter()
            .map(AnthropicSkillObject::from)
            .collect::<Vec<_>>(),
        Some(_) => Vec::new(),
    };

    let limit = query.and_then(|query| query.limit).unwrap_or(data.len());
    if limit < data.len() {
        data.truncate(limit);
    }
    let _ = query.and_then(|query| query.page.as_deref());

    AnthropicSkillListObject {
        data,
        has_more: false,
        next_page: None,
    }
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/skills",
    responses((status = 200, description = "Skill uploaded", body = SkillObject))
)]
pub async fn upload_skill(
    headers: HeaderMap,
    Extension(store): Extension<Arc<SkillStore>>,
    multipart: Multipart,
) -> axum::response::Response {
    match store.create_skill(multipart).await {
        Ok(skill) if prefers_anthropic_shape(&headers, None) => {
            Json(AnthropicSkillObject::from(&skill)).into_response()
        }
        Ok(skill) => Json(skill).into_response(),
        Err(e) => skill_error(StatusCode::UNPROCESSABLE_ENTITY, e),
    }
}

#[utoipa::path(
    get,
    tag = "Mistral.rs",
    path = "/v1/skills",
    responses((status = 200, description = "Uploaded skills", body = SkillListObject))
)]
pub async fn list_skills(
    headers: HeaderMap,
    Query(query): Query<SkillListQuery>,
    Extension(store): Extension<Arc<SkillStore>>,
) -> axum::response::Response {
    match store.list() {
        Ok(data) if prefers_anthropic_shape(&headers, Some(&query)) => {
            Json(anthropic_list_response(data, Some(&query))).into_response()
        }
        Ok(data) => Json(SkillListObject {
            object: "list",
            data,
        })
        .into_response(),
        Err(e) => skill_error(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/skills/{skill_id}/versions",
    responses((status = 200, description = "Skill version uploaded", body = SkillVersionObject))
)]
pub async fn upload_skill_version(
    AxumPath(skill_id): AxumPath<String>,
    headers: HeaderMap,
    Extension(store): Extension<Arc<SkillStore>>,
    multipart: Multipart,
) -> axum::response::Response {
    match store.create_version(&skill_id, multipart).await {
        Ok(version) if prefers_anthropic_shape(&headers, None) => {
            Json(AnthropicSkillVersionObject::from(&version)).into_response()
        }
        Ok(version) => Json(version).into_response(),
        Err(e) => skill_error(StatusCode::UNPROCESSABLE_ENTITY, e),
    }
}

#[utoipa::path(
    get,
    tag = "Mistral.rs",
    path = "/v1/skills/{skill_id}/versions",
    responses((status = 200, description = "Skill versions", body = AnthropicSkillVersionListObject))
)]
pub async fn list_skill_versions(
    AxumPath(skill_id): AxumPath<String>,
    Extension(store): Extension<Arc<SkillStore>>,
) -> axum::response::Response {
    match store.list_versions(&skill_id) {
        Ok(versions) => Json(AnthropicSkillVersionListObject {
            data: versions
                .iter()
                .map(AnthropicSkillVersionObject::from)
                .collect(),
            has_more: false,
            next_page: None,
        })
        .into_response(),
        Err(e) => skill_error(StatusCode::NOT_FOUND, e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_skill() -> SkillObject {
        SkillObject {
            id: "skill_abc".to_string(),
            object: SKILL_OBJECT,
            created_at: 1_700_000_000,
            name: "invoice-auditor".to_string(),
            description: "Checks invoices.".to_string(),
            latest_version: 2,
        }
    }

    #[test]
    fn anthropic_list_shape_filters_custom_skills() {
        let query = SkillListQuery {
            source: Some(CUSTOM_SKILL_SOURCE.to_string()),
            limit: None,
            page: None,
        };
        let response = anthropic_list_response(vec![test_skill()], Some(&query));

        assert!(!response.has_more);
        assert!(response.next_page.is_none());
        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0].id, "skill_abc");
        assert_eq!(response.data[0].tp, SKILL_OBJECT);
        assert_eq!(response.data[0].display_title, "invoice-auditor");
        assert_eq!(response.data[0].latest_version, "2");
        assert_eq!(response.data[0].source, CUSTOM_SKILL_SOURCE);
    }

    #[test]
    fn anthropic_list_shape_returns_empty_anthropic_source() {
        let query = SkillListQuery {
            source: Some(ANTHROPIC_SKILL_SOURCE.to_string()),
            limit: None,
            page: None,
        };
        let response = anthropic_list_response(vec![test_skill()], Some(&query));

        assert!(response.data.is_empty());
    }
}
