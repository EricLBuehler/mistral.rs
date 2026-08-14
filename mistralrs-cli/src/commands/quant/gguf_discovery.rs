use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
    sync::LazyLock,
};

use anyhow::{bail, Context, Result};
use mistralrs_core::{ModelDType, GGUF_MULTI_FILE_DELIMITER};
use regex::Regex;
use walkdir::WalkDir;

static SPLIT_GGUF_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?P<prefix>.+)-(?P<index>[0-9]+)-of-(?P<total>[0-9]+)\.gguf$")
        .expect("split GGUF regex is invalid")
});
static QUANT_SUFFIX_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(?:^|[-_.])(?P<label>(?:UD[-_])?(?:IQ[1-8](?:_[A-Z0-9]+)*|Q[1-8](?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP[0-9]+(?:_[A-Z0-9]+)*))$",
    )
    .expect("GGUF quant suffix regex is invalid")
});

const NUMERIC_QUANT_LEVELS: &[&str] = &["2", "3", "4", "5", "6", "8"];
const Q2_PREFERENCE: &[&str] = &["Q2K", "Q2KL", "Q2KS"];
const Q3_PREFERENCE: &[&str] = &["Q3KM", "Q3KS", "Q3KL", "Q3K"];
const Q4_PREFERENCE: &[&str] = &["Q4KM", "Q4KS", "Q4KL", "Q41", "Q40", "Q4K"];
const Q5_PREFERENCE: &[&str] = &["Q5KM", "Q5KS", "Q5KL", "Q51", "Q50", "Q5K"];
const Q4_K_PREFERENCE: &[&str] = &["Q4KM", "Q4KS", "Q4KL", "Q4K"];
const Q5_K_PREFERENCE: &[&str] = &["Q5KM", "Q5KS", "Q5KL", "Q5K"];
const Q6_PREFERENCE: &[&str] = &["Q6K", "Q6KS", "Q6KL"];
const Q8_PREFERENCE: &[&str] = &["Q80", "Q8K"];

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResolvedGgufArtifact {
    pub label: String,
    pub files: Vec<String>,
}

impl ResolvedGgufArtifact {
    pub fn file_spec(&self) -> String {
        self.files.join(GGUF_MULTI_FILE_DELIMITER)
    }
}

#[derive(Clone, Debug)]
struct GgufGroup {
    display: String,
    label: Option<String>,
    files: Vec<GroupFile>,
    split_totals: BTreeSet<usize>,
}

#[derive(Clone, Debug)]
struct GroupFile {
    path: String,
    index: Option<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ArtifactKind {
    Model,
    Projector,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum ProjectorRole {
    Vision,
    Audio,
}

pub(crate) fn has_gguf_model_files(files: &[String]) -> bool {
    files
        .iter()
        .any(|file| is_gguf(file) && !is_projector(file) && !is_auxiliary_gguf(file))
}

pub(crate) fn resolve_gguf_quant(
    files: &[String],
    requested: &str,
) -> Result<ResolvedGgufArtifact> {
    let groups = build_groups(files, ArtifactKind::Model);
    let preferences = quant_preferences(requested)?;
    let primary = groups
        .values()
        .filter(|group| !is_variant_artifact(group))
        .collect::<Vec<_>>();
    if let Some(group) = resolve_gguf_quant_groups(&primary, requested, &preferences)? {
        return Ok(group);
    }
    let variants = groups
        .values()
        .filter(|group| is_variant_artifact(group))
        .collect::<Vec<_>>();
    if let Some(group) = resolve_gguf_quant_groups(&variants, requested, &preferences)? {
        return Ok(group);
    }

    if let Some(bit_width) = preferences.bit_width {
        let iq_matches = groups
            .values()
            .filter(|group| {
                let Some(label) = group.label.as_deref().map(normalize_label) else {
                    return false;
                };
                is_iq_quant(&label) && quant_bits(&label) == Some(bit_width)
            })
            .collect::<Vec<_>>();
        if !iq_matches.is_empty() {
            bail!(
                "`--quant {requested}` only matched unsupported IQ GGUF artifacts: {}. Choose a \
                 supported Q/K quant",
                format_group_choices(&iq_matches)
            );
        }
    }

    let available = format_available_groups(groups.values());
    bail!("No GGUF model files matched `--quant {requested}`. Available GGUF variants: {available}")
}

fn resolve_gguf_quant_groups(
    groups: &[&GgufGroup],
    requested: &str,
    preferences: &QuantPreferences,
) -> Result<Option<ResolvedGgufArtifact>> {
    for label in &preferences.labels {
        let matches = groups
            .iter()
            .copied()
            .filter(|group| {
                group
                    .label
                    .as_deref()
                    .is_some_and(|candidate| normalize_label(candidate) == *label)
            })
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [] => {}
            [group] => return resolve_group(group).map(Some),
            _ => {
                bail!(
                    "GGUF quant `{requested}` is ambiguous: {}",
                    format_group_choices(&matches)
                )
            }
        }
    }

    if let Some(bit_width) = preferences.bit_width {
        for label in &preferences.labels {
            let ud_label = format!("UD{label}");
            let matches = groups
                .iter()
                .copied()
                .filter(|group| {
                    group
                        .label
                        .as_deref()
                        .is_some_and(|candidate| normalize_label(candidate) == ud_label)
                })
                .collect::<Vec<_>>();
            match matches.as_slice() {
                [] => {}
                [group] => return resolve_group(group).map(Some),
                _ => {
                    bail!(
                        "GGUF quant `{requested}` is ambiguous: {}",
                        format_group_choices(&matches)
                    )
                }
            }
        }

        let mut fallback = groups
            .iter()
            .copied()
            .filter_map(|group| {
                let label = normalize_label(group.label.as_deref()?);
                (!is_iq_quant(&label) && quant_bits(&label) == Some(bit_width))
                    .then_some((quant_fallback_rank(&label), group))
            })
            .collect::<Vec<_>>();
        fallback.sort_by_key(|(rank, group)| (*rank, group.display.to_ascii_lowercase()));
        if let Some((best_rank, _)) = fallback.first() {
            let best = fallback
                .iter()
                .take_while(|(rank, _)| rank == best_rank)
                .map(|(_, group)| *group)
                .collect::<Vec<_>>();
            match best.as_slice() {
                [group] => return resolve_group(group).map(Some),
                [] => unreachable!("nonempty fallback has a best group"),
                _ => {
                    bail!(
                        "GGUF quant `{requested}` has multiple equally ranked fallbacks: {}. Pass \
                         an explicit quant name",
                        format_group_choices(&best)
                    )
                }
            }
        }
    }

    Ok(None)
}

pub(crate) fn resolve_gguf_projector(
    files: &[String],
    dtype: ModelDType,
) -> Result<Option<ResolvedGgufArtifact>> {
    let groups = build_groups(files, ArtifactKind::Projector);
    if groups.is_empty() {
        return Ok(None);
    }

    let unsupported_iq = groups
        .values()
        .filter(|group| is_iq_projector(group))
        .collect::<Vec<_>>();
    let supported = groups
        .values()
        .filter(|group| !is_iq_projector(group))
        .collect::<Vec<_>>();
    if supported.is_empty() {
        bail!(
            "Automatic GGUF projector selection only found unsupported IQ artifacts: {}. Choose \
             a BF16, F16, F32, or supported Q/K projector",
            format_group_choices(&unsupported_iq)
        );
    }

    let role_groups = supported
        .iter()
        .copied()
        .filter_map(|group| projector_role(group).map(|role| (role, group)))
        .collect::<Vec<_>>();
    if !role_groups.is_empty() {
        if role_groups.len() != supported.len() {
            bail!(
                "Automatic GGUF projector selection cannot mix role-specific and generic \
                 projectors: {}. Pass `--mmproj` to choose explicitly",
                format_group_choices(&supported)
            );
        }

        let mut selected = Vec::new();
        for role in [ProjectorRole::Vision, ProjectorRole::Audio] {
            let candidates = role_groups
                .iter()
                .filter_map(|(candidate_role, group)| (*candidate_role == role).then_some(*group))
                .collect::<Vec<_>>();
            if candidates.is_empty() {
                let unsupported = unsupported_iq
                    .iter()
                    .copied()
                    .filter(|group| projector_role(group) == Some(role))
                    .collect::<Vec<_>>();
                if !unsupported.is_empty() {
                    let role = match role {
                        ProjectorRole::Vision => "vision",
                        ProjectorRole::Audio => "audio",
                    };
                    bail!(
                        "Automatic GGUF projector selection only found unsupported IQ artifacts \
                         for the {role} projector: {}. Choose a BF16, F16, F32, or supported Q/K \
                         projector",
                        format_group_choices(&unsupported)
                    );
                }
                continue;
            }
            selected.push((role, select_projector_group(&candidates, dtype)?));
        }

        let mut files = Vec::new();
        let mut labels = Vec::new();
        for (role, group) in selected {
            let artifact = resolve_group(group)?;
            let role = match role {
                ProjectorRole::Vision => "vision",
                ProjectorRole::Audio => "audio",
            };
            labels.push(format!("{role} {}", artifact.label));
            files.extend(artifact.files);
        }
        return Ok(Some(ResolvedGgufArtifact {
            label: labels.join(" + "),
            files,
        }));
    }

    let unsupported_role_groups = unsupported_iq
        .iter()
        .copied()
        .filter(|group| projector_role(group).is_some())
        .collect::<Vec<_>>();
    if !unsupported_role_groups.is_empty() {
        bail!(
            "Automatic GGUF projector selection cannot mix a generic projector with unsupported \
             role-specific IQ artifacts: {}. Pass `--mmproj` to choose explicitly",
            format_group_choices(&unsupported_role_groups)
        );
    }

    resolve_group(select_projector_group(&supported, dtype)?).map(Some)
}

fn select_projector_group<'a>(
    groups: &[&'a GgufGroup],
    dtype: ModelDType,
) -> Result<&'a GgufGroup> {
    let mut ranked = groups
        .iter()
        .map(|group| (projector_rank(group.label.as_deref(), dtype), *group))
        .collect::<Vec<_>>();
    ranked.sort_by_key(|(rank, group)| (*rank, group.display.to_ascii_lowercase()));
    let best_rank = ranked[0].0;
    let best = ranked
        .iter()
        .take_while(|(rank, _)| *rank == best_rank)
        .map(|(_, group)| *group)
        .collect::<Vec<_>>();
    if best.len() != 1 {
        bail!(
            "Automatic GGUF projector selection is ambiguous: {}. Pass `--mmproj` to choose one",
            format_group_choices(&best)
        );
    }
    Ok(best[0])
}

pub(crate) fn list_local_files_recursive(root: &Path) -> Result<Vec<String>> {
    if !root.is_dir() {
        return Ok(Vec::new());
    }

    let mut files = Vec::new();
    for entry in WalkDir::new(root).follow_links(false).min_depth(1) {
        let entry =
            entry.with_context(|| format!("Cannot recursively list `{}`", root.display()))?;
        if !entry.file_type().is_file() && !entry.file_type().is_symlink() {
            continue;
        }
        let relative = entry
            .path()
            .strip_prefix(root)
            .with_context(|| format!("Cannot make `{}` relative", entry.path().display()))?;
        files.push(relative.to_string_lossy().replace('\\', "/"));
    }
    files.sort();
    Ok(files)
}

pub(crate) fn list_local_gguf_companions(root: &Path, exact_file: &str) -> Result<Vec<String>> {
    let mut directories = BTreeSet::from([root.to_path_buf()]);
    for filename in exact_file.split(GGUF_MULTI_FILE_DELIMITER).map(str::trim) {
        let path = Path::new(filename);
        let resolved = if path.is_absolute() {
            path.to_path_buf()
        } else {
            root.join(path)
        };
        if let Some(parent) = resolved.parent() {
            directories.insert(parent.to_path_buf());
        }
    }

    let mut files = BTreeSet::new();
    for directory in directories {
        if !directory.is_dir() {
            continue;
        }
        for entry in fs::read_dir(&directory).with_context(|| {
            format!("Cannot list local GGUF directory `{}`", directory.display())
        })? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if !file_type.is_file() && !file_type.is_symlink() {
                continue;
            }
            let path = entry.path();
            files.insert(
                path.strip_prefix(root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
        }
    }
    Ok(files.into_iter().collect())
}

fn build_groups(files: &[String], kind: ArtifactKind) -> BTreeMap<String, GgufGroup> {
    let mut groups = BTreeMap::new();
    for original in files {
        let path = original.replace('\\', "/");
        if !is_gguf(&path) || is_auxiliary_gguf(&path) {
            continue;
        }
        let projector = is_projector(&path);
        if projector != matches!(kind, ArtifactKind::Projector) {
            continue;
        }

        if let Some(split) = parse_split(&path) {
            let label = quant_label(&split.prefix);
            if matches!(kind, ArtifactKind::Model) && label.is_none() {
                continue;
            }
            let key = format!("split:{}", split.prefix);
            let group = groups.entry(key).or_insert_with(|| GgufGroup {
                display: split.prefix.clone(),
                label,
                files: Vec::new(),
                split_totals: BTreeSet::new(),
            });
            group.files.push(GroupFile {
                path,
                index: Some(split.index),
            });
            group.split_totals.insert(split.total);
        } else {
            let Some(stem) = logical_stem(&path) else {
                continue;
            };
            let label = quant_label(&stem);
            if matches!(kind, ArtifactKind::Model) && label.is_none() {
                continue;
            }
            groups.insert(
                format!("single:{path}"),
                GgufGroup {
                    display: path.clone(),
                    label,
                    files: vec![GroupFile { path, index: None }],
                    split_totals: BTreeSet::new(),
                },
            );
        }
    }
    groups
}

fn resolve_group(group: &GgufGroup) -> Result<ResolvedGgufArtifact> {
    let mut files = group.files.clone();
    if !group.split_totals.is_empty() {
        if group.split_totals.len() != 1 {
            bail!(
                "GGUF shard group `{}` declares conflicting shard totals: {:?}",
                group.display,
                group.split_totals
            );
        }
        let total = *group
            .split_totals
            .first()
            .expect("nonempty shard totals have a first element");
        if total == 0 {
            bail!("GGUF shard group `{}` declares zero shards", group.display);
        }

        let mut indices = BTreeSet::new();
        for file in &files {
            let index = file.index.expect("split GGUF file has an index");
            if index == 0 || index > total {
                bail!(
                    "GGUF shard `{}` has index {index}, expected a value from 1 through {total}",
                    file.path
                );
            }
            if !indices.insert(index) {
                bail!(
                    "GGUF shard group `{}` contains duplicate shard {index}",
                    group.display
                );
            }
        }
        let missing = (1..=total)
            .filter(|index| !indices.contains(index))
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            bail!(
                "GGUF shard group `{}` is incomplete: missing shard indices {} of {total}",
                group.display,
                missing
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        files.sort_by_key(|file| file.index);
    }

    Ok(ResolvedGgufArtifact {
        label: group.label.clone().unwrap_or_else(|| "unknown".to_string()),
        files: files.into_iter().map(|file| file.path).collect(),
    })
}

struct QuantPreferences {
    labels: Vec<String>,
    bit_width: Option<usize>,
}

fn quant_preferences(requested: &str) -> Result<QuantPreferences> {
    let lowered = requested.trim().to_ascii_lowercase();
    if lowered.is_empty() {
        bail!("GGUF quant selection cannot be empty");
    }
    if lowered == "auto" {
        bail!("`--quant auto` does not select a GGUF artifact; choose a bit width or quant name");
    }
    if is_iq_quant(&normalize_label(&lowered)) {
        bail!("IQ GGUF formats are not supported; choose a supported Q/K quant");
    }
    if lowered.chars().all(|ch| ch.is_ascii_digit())
        && !NUMERIC_QUANT_LEVELS.contains(&lowered.as_str())
    {
        bail!(
            "GGUF numeric quant selection must be one of {}",
            NUMERIC_QUANT_LEVELS.join(", ")
        );
    }

    let (preference, bit_width) = match lowered.as_str() {
        "2" | "q2" => (Q2_PREFERENCE, Some(2)),
        "3" | "q3" => (Q3_PREFERENCE, Some(3)),
        "4" | "q4" => (Q4_PREFERENCE, Some(4)),
        "5" | "q5" => (Q5_PREFERENCE, Some(5)),
        "6" | "q6" => (Q6_PREFERENCE, Some(6)),
        "8" | "q8" => (Q8_PREFERENCE, Some(8)),
        "q2k" => (Q2_PREFERENCE, None),
        "q3k" => (Q3_PREFERENCE, None),
        "q4k" => (Q4_K_PREFERENCE, None),
        "q5k" => (Q5_K_PREFERENCE, None),
        "q6k" => (Q6_PREFERENCE, None),
        other => {
            return Ok(QuantPreferences {
                labels: vec![normalize_label(other)],
                bit_width: None,
            })
        }
    };
    Ok(QuantPreferences {
        labels: preference
            .iter()
            .map(|label| (*label).to_string())
            .collect(),
        bit_width,
    })
}

fn projector_rank(label: Option<&str>, dtype: ModelDType) -> usize {
    let float_preference = match dtype {
        ModelDType::Auto | ModelDType::BF16 => ["BF16", "F16", "F32"],
        ModelDType::F16 => ["F16", "BF16", "F32"],
        ModelDType::F32 => ["F32", "BF16", "F16"],
    };
    let Some(label) = label else {
        return float_preference.len();
    };
    let normalized = normalize_label(label);
    if let Some(rank) = float_preference
        .iter()
        .position(|candidate| *candidate == normalized)
    {
        return rank;
    }

    let bits = quant_bits(&normalized).unwrap_or(0);
    100 + (8usize.saturating_sub(bits) * 10)
}

fn quant_bits(normalized: &str) -> Option<usize> {
    let label = normalized.strip_prefix("UD").unwrap_or(normalized);
    let label = label
        .strip_prefix("IQ")
        .or_else(|| label.strip_prefix('Q'))
        .or_else(|| label.strip_prefix("MXFP"))?;
    let bit = label.chars().next()?;
    bit.to_digit(10).map(|bit| bit as usize)
}

fn is_iq_quant(normalized: &str) -> bool {
    normalized
        .strip_prefix("UD")
        .unwrap_or(normalized)
        .starts_with("IQ")
}

fn is_iq_projector(group: &GgufGroup) -> bool {
    group
        .label
        .as_deref()
        .map(normalize_label)
        .is_some_and(|label| is_iq_quant(&label))
}

fn quant_fallback_rank(normalized: &str) -> usize {
    if normalized.starts_with('Q') {
        0
    } else if normalized.starts_with("UDQ") {
        1
    } else if normalized.starts_with("MXFP") {
        2
    } else {
        3
    }
}

fn format_available_groups<'a>(groups: impl Iterator<Item = &'a GgufGroup>) -> String {
    let mut available = groups
        .map(|group| {
            format!(
                "{} ({})",
                group.label.as_deref().unwrap_or("unknown"),
                group.display
            )
        })
        .collect::<Vec<_>>();
    available.sort_by_key(|entry| entry.to_ascii_lowercase());
    if available.is_empty() {
        "none".to_string()
    } else {
        available.join(", ")
    }
}

fn format_group_choices(groups: &[&GgufGroup]) -> String {
    let mut choices = groups
        .iter()
        .map(|group| group.display.clone())
        .collect::<Vec<_>>();
    choices.sort_by_key(|choice| choice.to_ascii_lowercase());
    choices.join(", ")
}

fn quant_label(stem: &str) -> Option<String> {
    stem.rsplit('/').find_map(|name| {
        QUANT_SUFFIX_RE
            .captures(name)
            .and_then(|captures| captures.name("label"))
            .map(|label| canonical_label(label.as_str()))
    })
}

fn canonical_label(label: &str) -> String {
    let label = label.to_ascii_uppercase();
    label
        .strip_prefix("UD_")
        .map_or(label.clone(), |rest| format!("UD-{rest}"))
}

fn normalize_label(label: &str) -> String {
    label
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_uppercase)
        .collect()
}

fn logical_stem(path: &str) -> Option<String> {
    if let Some(split) = parse_split(path) {
        return Some(split.prefix);
    }
    let extension_start = path.len().checked_sub(".gguf".len())?;
    path.get(extension_start..)
        .is_some_and(|extension| extension.eq_ignore_ascii_case(".gguf"))
        .then(|| path[..extension_start].to_string())
}

fn is_gguf(path: &str) -> bool {
    path.get(path.len().saturating_sub(".gguf".len())..)
        .is_some_and(|extension| extension.eq_ignore_ascii_case(".gguf"))
}

fn is_projector(path: &str) -> bool {
    has_filename_token(path, "mmproj")
}

fn projector_role(group: &GgufGroup) -> Option<ProjectorRole> {
    let vision = has_path_token(&group.display, "vision");
    let audio = has_path_token(&group.display, "audio");
    match (vision, audio) {
        (true, false) => Some(ProjectorRole::Vision),
        (false, true) => Some(ProjectorRole::Audio),
        _ => None,
    }
}

fn is_auxiliary_gguf(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    let name = lower.rsplit('/').next().unwrap_or(&lower);
    name.starts_with("mtp-")
        || name.starts_with("mtp_")
        || name.starts_with("dflash-")
        || name.starts_with("dflash_")
        || name.starts_with("lora-")
        || name.starts_with("lora_")
        || name.starts_with("imatrix-")
        || name.starts_with("imatrix_")
        || name == "imatrix.gguf"
        || has_filename_token(path, "dflash")
        || lower.split('/').any(|component| {
            component == "mtp"
                || component == "dflash"
                || component == "lora"
                || component == "imatrix"
        })
}

fn is_variant_artifact(group: &GgufGroup) -> bool {
    group
        .files
        .iter()
        .any(|file| has_filename_token(&file.path, "mtp") || has_filename_token(&file.path, "lora"))
}

fn has_filename_token(path: &str, token: &str) -> bool {
    path.rsplit('/').next().is_some_and(|name| {
        name.split(|ch: char| !ch.is_ascii_alphanumeric())
            .any(|component| component.eq_ignore_ascii_case(token))
    })
}

fn has_path_token(path: &str, token: &str) -> bool {
    path.split('/').any(|name| {
        name.split(|ch: char| !ch.is_ascii_alphanumeric())
            .any(|component| component.eq_ignore_ascii_case(token))
    })
}

struct SplitGguf {
    prefix: String,
    index: usize,
    total: usize,
}

fn parse_split(path: &str) -> Option<SplitGguf> {
    let captures = SPLIT_GGUF_RE.captures(path)?;
    let prefix = captures.name("prefix")?.as_str();
    let split_suffix_start = prefix.len().saturating_sub("-split".len());
    let prefix = if prefix
        .get(split_suffix_start..)
        .is_some_and(|suffix| suffix.eq_ignore_ascii_case("-split"))
    {
        &prefix[..split_suffix_start]
    } else {
        prefix
    };
    Some(SplitGguf {
        prefix: prefix.to_string(),
        index: captures.name("index")?.as_str().parse().ok()?,
        total: captures.name("total")?.as_str().parse().ok()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(entries: &[&str]) -> Vec<String> {
        entries.iter().map(|entry| (*entry).to_string()).collect()
    }

    #[test]
    fn numeric_q4_prefers_canonical_medium() {
        let listing = files(&[
            "model-IQ4_XS.gguf",
            "model-Q4_0.gguf",
            "model-Q4_K_S.gguf",
            "model-Q4_K_M.gguf",
            "model-UD-Q4_K_XL.gguf",
            "mmproj-BF16.gguf",
        ]);
        let selected = resolve_gguf_quant(&listing, "4").unwrap();
        assert_eq!(selected.label, "Q4_K_M");
        assert_eq!(selected.files, files(&["model-Q4_K_M.gguf"]));
    }

    #[test]
    fn numeric_q4_uses_deterministic_canonical_fallback() {
        let listing = files(&["model-Q4_0.gguf", "model-Q4_1.gguf", "model-Q4_K_S.gguf"]);
        assert_eq!(resolve_gguf_quant(&listing, "4").unwrap().label, "Q4_K_S");
    }

    #[test]
    fn explicit_iq_is_rejected_and_ud_names_do_not_cross_families() {
        let listing = files(&[
            "model-IQ4_NL.gguf",
            "model-IQ4_XS.gguf",
            "model-Q4_K_M.gguf",
            "model-UD-Q4_K_XL.gguf",
        ]);
        let error = resolve_gguf_quant(&listing, "iq4_xs").unwrap_err();
        assert!(error
            .to_string()
            .contains("IQ GGUF formats are not supported"));
        assert_eq!(
            resolve_gguf_quant(&listing, "ud-q4_k_xl").unwrap().label,
            "UD-Q4_K_XL"
        );
    }

    #[test]
    fn numeric_width_does_not_fall_back_to_iq() {
        let listing = files(&["model-IQ4_NL.gguf", "model-IQ4_XS.gguf"]);
        let error = resolve_gguf_quant(&listing, "4").unwrap_err();
        assert!(error
            .to_string()
            .contains("only matched unsupported IQ GGUF artifacts"));
    }

    #[test]
    fn explicit_k_family_does_not_fall_back_to_a_different_family() {
        let listing = files(&["model-Q4_0.gguf", "model-IQ4_XS.gguf"]);
        let error = resolve_gguf_quant(&listing, "q4k").unwrap_err();
        assert!(error.to_string().contains("No GGUF model files matched"));
    }

    #[test]
    fn explicit_k_family_accepts_an_unsuffixed_k_quant() {
        let listing = files(&["model-Q4_K.gguf", "model-Q4_0.gguf"]);
        assert_eq!(resolve_gguf_quant(&listing, "q4k").unwrap().label, "Q4_K");
    }

    #[test]
    fn nested_shards_are_expanded_in_index_order() {
        let listing = files(&[
            "Q4_K_M/model-Q4_K_M-00003-of-00003.gguf",
            "Q4_K_M/model-Q4_K_M-00001-of-00003.gguf",
            "Q4_K_M/model-Q4_K_M-00002-of-00003.gguf",
        ]);
        let selected = resolve_gguf_quant(&listing, "4").unwrap();
        assert_eq!(
            selected.files,
            files(&[
                "Q4_K_M/model-Q4_K_M-00001-of-00003.gguf",
                "Q4_K_M/model-Q4_K_M-00002-of-00003.gguf",
                "Q4_K_M/model-Q4_K_M-00003-of-00003.gguf",
            ])
        );
        assert_eq!(
            selected.file_spec(),
            selected.files.join(GGUF_MULTI_FILE_DELIMITER)
        );
    }

    #[test]
    fn split_marker_and_quant_directory_are_supported() {
        let split_marker = files(&[
            "BF16/model-BF16-split-00002-of-00002.gguf",
            "BF16/model-BF16-split-00001-of-00002.gguf",
        ]);
        assert_eq!(
            resolve_gguf_quant(&split_marker, "bf16").unwrap().files,
            files(&[
                "BF16/model-BF16-split-00001-of-00002.gguf",
                "BF16/model-BF16-split-00002-of-00002.gguf",
            ])
        );

        let directory_only = files(&[
            "Q4_K_M/model-00002-of-00002.gguf",
            "Q4_K_M/model-00001-of-00002.gguf",
        ]);
        assert_eq!(
            resolve_gguf_quant(&directory_only, "4").unwrap().files,
            files(&[
                "Q4_K_M/model-00001-of-00002.gguf",
                "Q4_K_M/model-00002-of-00002.gguf",
            ])
        );
    }

    #[test]
    fn selected_shard_group_must_be_complete() {
        let listing = files(&[
            "Q4_K_M/model-Q4_K_M-00001-of-00003.gguf",
            "Q4_K_M/model-Q4_K_M-00003-of-00003.gguf",
        ]);
        let error = resolve_gguf_quant(&listing, "4").unwrap_err();
        assert!(error.to_string().contains("missing shard indices 2 of 3"));
    }

    #[test]
    fn equal_rank_model_groups_are_ambiguous() {
        let listing = files(&["one/model-a-Q4_K_M.gguf", "two/model-b-Q4_K_M.gguf"]);
        let error = resolve_gguf_quant(&listing, "4").unwrap_err();
        assert!(error.to_string().contains("ambiguous"));
        assert!(error.to_string().contains("one/model-a-Q4_K_M.gguf"));
        assert!(error.to_string().contains("two/model-b-Q4_K_M.gguf"));
    }

    #[test]
    fn case_distinct_artifacts_are_not_silently_merged() {
        let listing = files(&["Model-Q4_K_M.gguf", "model-Q4_K_M.gguf"]);
        let error = resolve_gguf_quant(&listing, "4").unwrap_err();
        assert!(error.to_string().contains("ambiguous"));
        assert!(error.to_string().contains("Model-Q4_K_M.gguf"));
        assert!(error.to_string().contains("model-Q4_K_M.gguf"));
    }

    #[test]
    fn projector_float_choice_tracks_dtype() {
        let listing = files(&["mmproj-F32.gguf", "mmproj-BF16.gguf", "mmproj-F16.gguf"]);
        assert_eq!(
            resolve_gguf_projector(&listing, ModelDType::Auto)
                .unwrap()
                .unwrap()
                .label,
            "BF16"
        );
        assert_eq!(
            resolve_gguf_projector(&listing, ModelDType::F16)
                .unwrap()
                .unwrap()
                .label,
            "F16"
        );
        assert_eq!(
            resolve_gguf_projector(&listing, ModelDType::F32)
                .unwrap()
                .unwrap()
                .label,
            "F32"
        );
    }

    #[test]
    fn float_projector_is_preferred_over_quantized_projector() {
        let listing = files(&["mmproj-Q8_0.gguf", "mmproj-F32.gguf"]);
        assert_eq!(
            resolve_gguf_projector(&listing, ModelDType::BF16)
                .unwrap()
                .unwrap()
                .label,
            "F32"
        );
    }

    #[test]
    fn infix_projector_names_are_discovered() {
        let listing = files(&[
            "Bonsai-27B-Q4_K_M.gguf",
            "Bonsai-27B-mmproj-BF16.gguf",
            "Bonsai-27B-mmproj-Q8_0.gguf",
        ]);
        assert_eq!(
            resolve_gguf_quant(&listing, "4").unwrap().files,
            files(&["Bonsai-27B-Q4_K_M.gguf"])
        );
        assert_eq!(
            resolve_gguf_projector(&listing, ModelDType::Auto)
                .unwrap()
                .unwrap()
                .files,
            files(&["Bonsai-27B-mmproj-BF16.gguf"])
        );
    }

    #[test]
    fn highest_bit_quantized_projector_is_preferred() {
        let listing = files(&["mmproj-Q4_K_M.gguf", "mmproj-Q8_0.gguf"]);
        assert_eq!(
            resolve_gguf_projector(&listing, ModelDType::Auto)
                .unwrap()
                .unwrap()
                .label,
            "Q8_0"
        );
    }

    #[test]
    fn iq_only_projector_is_rejected() {
        let listing = files(&["mmproj-IQ4_XS.gguf"]);
        let error = resolve_gguf_projector(&listing, ModelDType::Auto).unwrap_err();

        assert!(error.to_string().contains("unsupported IQ artifacts"));
        assert!(error.to_string().contains("mmproj-IQ4_XS.gguf"));
    }

    #[test]
    fn supported_projector_is_preferred_over_iq_projector() {
        let listing = files(&["mmproj-IQ4_XS.gguf", "mmproj-Q4_K_M.gguf"]);
        let selected = resolve_gguf_projector(&listing, ModelDType::Auto)
            .unwrap()
            .unwrap();

        assert_eq!(selected.label, "Q4_K_M");
        assert_eq!(selected.files, files(&["mmproj-Q4_K_M.gguf"]));
    }

    #[test]
    fn independent_vision_and_audio_projectors_are_selected() {
        let listing = files(&[
            "model-vision-mmproj-F32.gguf",
            "model-vision-mmproj-F16.gguf",
            "model-audio-mmproj-Q8_0.gguf",
            "model-audio-mmproj-BF16.gguf",
        ]);
        let selected = resolve_gguf_projector(&listing, ModelDType::Auto)
            .unwrap()
            .unwrap();

        assert_eq!(selected.label, "vision F16 + audio BF16");
        assert_eq!(
            selected.files,
            files(&[
                "model-vision-mmproj-F16.gguf",
                "model-audio-mmproj-BF16.gguf",
            ])
        );
    }

    #[test]
    fn role_specific_projectors_ignore_iq_candidates() {
        let listing = files(&[
            "vision-mmproj-IQ4_XS.gguf",
            "vision-mmproj-Q4_K_M.gguf",
            "audio-mmproj-IQ4_XS.gguf",
            "audio-mmproj-BF16.gguf",
        ]);
        let selected = resolve_gguf_projector(&listing, ModelDType::Auto)
            .unwrap()
            .unwrap();

        assert_eq!(selected.label, "vision Q4_K_M + audio BF16");
        assert_eq!(
            selected.files,
            files(&["vision-mmproj-Q4_K_M.gguf", "audio-mmproj-BF16.gguf",])
        );
    }

    #[test]
    fn role_with_only_iq_projectors_is_rejected() {
        let listing = files(&["vision-mmproj-Q4_K_M.gguf", "audio-mmproj-IQ4_XS.gguf"]);
        let error = resolve_gguf_projector(&listing, ModelDType::Auto).unwrap_err();

        assert!(error
            .to_string()
            .contains("unsupported IQ artifacts for the audio projector"));
        assert!(error.to_string().contains("audio-mmproj-IQ4_XS.gguf"));
    }

    #[test]
    fn generic_projector_does_not_hide_role_specific_iq_projectors() {
        let listing = files(&["mmproj-BF16.gguf", "audio-mmproj-IQ4_XS.gguf"]);
        let error = resolve_gguf_projector(&listing, ModelDType::Auto).unwrap_err();

        assert!(error.to_string().contains(
            "cannot mix a generic projector with unsupported role-specific IQ artifacts"
        ));
        assert!(error.to_string().contains("audio-mmproj-IQ4_XS.gguf"));
    }

    #[test]
    fn independent_projector_shards_remain_grouped() {
        let listing = files(&[
            "vision-mmproj-BF16-00002-of-00002.gguf",
            "audio-mmproj-BF16.gguf",
            "vision-mmproj-BF16-00001-of-00002.gguf",
        ]);
        let selected = resolve_gguf_projector(&listing, ModelDType::Auto)
            .unwrap()
            .unwrap();

        assert_eq!(
            selected.files,
            files(&[
                "vision-mmproj-BF16-00001-of-00002.gguf",
                "vision-mmproj-BF16-00002-of-00002.gguf",
                "audio-mmproj-BF16.gguf",
            ])
        );
    }

    #[test]
    fn generic_and_role_specific_projectors_require_an_override() {
        let listing = files(&["mmproj-BF16.gguf", "audio-mmproj-BF16.gguf"]);
        let error = resolve_gguf_projector(&listing, ModelDType::Auto).unwrap_err();
        assert!(error.to_string().contains("role-specific and generic"));
        assert!(error.to_string().contains("--mmproj"));
    }

    #[test]
    fn equal_rank_projectors_require_an_override() {
        let listing = files(&["first/mmproj-BF16.gguf", "second/mmproj-BF16.gguf"]);
        let error = resolve_gguf_projector(&listing, ModelDType::Auto).unwrap_err();
        assert!(error.to_string().contains("ambiguous"));
        assert!(error.to_string().contains("--mmproj"));
    }

    #[test]
    fn auxiliary_ggufs_are_not_model_candidates() {
        let listing = files(&[
            "MTP/mtp-model-Q4_K_M.gguf",
            "lora-model-Q4_K_M.gguf",
            "model-MTP-Q4_K_M.gguf",
            "model-LoRA-Q4_K_M.gguf",
            "model-Q4_K_S.gguf",
        ]);
        assert_eq!(resolve_gguf_quant(&listing, "4").unwrap().label, "Q4_K_S");
    }

    #[test]
    fn infix_mtp_model_is_used_when_no_primary_variant_matches() {
        let listing = files(&[
            "Qwopus-Coder-MTP-Q4_K_M.gguf",
            "Qwopus-Coder-IMatrix-Q5_K_M.gguf",
            "mmproj-F32.gguf",
        ]);
        assert_eq!(
            resolve_gguf_quant(&listing, "4").unwrap().files,
            files(&["Qwopus-Coder-MTP-Q4_K_M.gguf"])
        );
        assert_eq!(
            resolve_gguf_quant(&listing, "5").unwrap().files,
            files(&["Qwopus-Coder-IMatrix-Q5_K_M.gguf"])
        );
    }

    #[test]
    fn numeric_width_falls_back_to_ud_and_mxfp_variants() {
        let ud_listing = files(&[
            "model-UD-IQ2_M.gguf",
            "model-UD-IQ2_XXS.gguf",
            "model-UD-Q2_K_XL.gguf",
        ]);
        assert_eq!(
            resolve_gguf_quant(&ud_listing, "2").unwrap().label,
            "UD-Q2_K_XL"
        );

        let mxfp_listing = files(&["model-MXFP4_MOE.gguf", "model-Q8_0.gguf"]);
        assert_eq!(
            resolve_gguf_quant(&mxfp_listing, "4").unwrap().label,
            "MXFP4_MOE"
        );
    }

    #[test]
    fn equally_ranked_numeric_fallbacks_require_an_explicit_name() {
        let listing = files(&["model-Q2_K_P.gguf", "model-Q2_G64.gguf"]);
        let error = resolve_gguf_quant(&listing, "2").unwrap_err();
        assert!(error.to_string().contains("equally ranked fallbacks"));
        assert!(error.to_string().contains("explicit quant name"));
    }

    #[test]
    fn unsupported_numeric_width_is_rejected() {
        let error = resolve_gguf_quant(&files(&["model-Q4_K_M.gguf"]), "7").unwrap_err();
        assert!(error.to_string().contains("2, 3, 4, 5, 6, 8"));
    }
}
