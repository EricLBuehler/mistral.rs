use std::{
    cmp::Reverse,
    collections::{BTreeMap, HashMap, HashSet},
    io::{self, Write},
    ops::Range,
    path::PathBuf,
};

use anyhow::{anyhow, Result};
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    style::{Color, ResetColor, SetBackgroundColor, SetForegroundColor},
    terminal::{self, ClearType},
};
use fuzzy_matcher::{skim::SkimMatcherV2, FuzzyMatcher};
use mistralrs_core::{
    expand_uqff_shards, list_model_files, read_model_file_range, resolve_uqff_shorthand,
    try_get_model_file, TokenSource,
};
use mistralrs_quant::{
    build_uqff_report_from_artifacts, inspect_uqff_artifacts, verify_uqff_artifacts,
    write_uqff_report, QuantizedSerdeType, UqffArtifactFile, UqffArtifactGroup, UqffArtifacts,
    UqffGeneratedBy, UqffInspection, UqffMetadataSummary, UqffReport, UqffReportOptions,
    UqffTensorSummary, UqffVerifyOptions, UQFF_REPORT_JSON,
};

use crate::args::{GlobalOptions, UqffCommand};

const DEFAULT_REVISION: &str = "main";

pub async fn run_uqff(command: UqffCommand, global: GlobalOptions) -> Result<()> {
    match command {
        UqffCommand::Report {
            model_id,
            quant,
            revision,
            write,
            json,
            base_model,
            repo_id,
        } => {
            run_report(
                ReportCommandArgs {
                    model_id,
                    quant,
                    revision,
                    write,
                    json,
                    verbose: global.verbose > 0,
                    base_model,
                    repo_id,
                },
                &global.token_source,
            )
            .await
        }
        UqffCommand::Verify {
            model_id,
            quant,
            revision,
            json,
            strict,
            allow_newer_minor,
        } => {
            run_verify(
                model_id,
                quant,
                revision,
                json,
                strict,
                allow_newer_minor,
                &global.token_source,
            )
            .await
        }
        UqffCommand::Inspect {
            model_id,
            quant,
            revision,
        } => run_inspect(model_id, quant, revision, &global.token_source).await,
    }
}

struct ReportCommandArgs {
    model_id: String,
    quant: Option<String>,
    revision: Option<String>,
    write: bool,
    json: bool,
    verbose: bool,
    base_model: Option<String>,
    repo_id: Option<String>,
}

async fn run_report(args: ReportCommandArgs, token_source: &TokenSource) -> Result<()> {
    let ReportCommandArgs {
        model_id,
        quant,
        revision,
        write,
        json,
        verbose,
        base_model,
        repo_id,
    } = args;
    let resolved = resolve_uqff_artifacts(
        &model_id,
        revision.as_deref(),
        quant.as_deref(),
        token_source,
    )
    .await?;
    let report = build_uqff_report_from_artifacts(
        &resolved.artifacts,
        UqffReportOptions {
            generated_by: generated_by("mistralrs uqff report"),
            base_model,
            repo_id,
        },
    )
    .await
    .map_err(|e| anyhow!("{e}"))?;

    if write {
        let Some(path) = resolved.local_write_path else {
            return Err(anyhow!(
                "`mistralrs uqff report --write` requires a local model path. Use `--json` for Hugging Face repos."
            ));
        };
        let report_path = write_uqff_report(&path, &report).map_err(|e| anyhow!("{e}"))?;
        eprintln!("Wrote {}", report_path.display());
    }

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_report_summary(&report, verbose);
    }

    Ok(())
}

async fn run_verify(
    model_id: String,
    quant: Option<String>,
    revision: Option<String>,
    json: bool,
    strict: bool,
    allow_newer_minor: bool,
    token_source: &TokenSource,
) -> Result<()> {
    let resolved = resolve_uqff_artifacts(
        &model_id,
        revision.as_deref(),
        quant.as_deref(),
        token_source,
    )
    .await?;
    let result = verify_uqff_artifacts(
        &resolved.artifacts,
        UqffVerifyOptions {
            strict,
            allow_newer_minor,
        },
    )
    .await
    .map_err(|e| anyhow!("{e}"))?;

    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        if result.ok {
            println!("OK: UQFF artifacts are structurally valid.");
        } else {
            println!("FAIL: UQFF verification failed.");
        }
        print_report_summary(&result.report, false);
        for warning in &result.warnings {
            println!("warning: {warning}");
        }
        for error in &result.errors {
            println!("error: {error}");
        }
    }

    if result.ok {
        Ok(())
    } else {
        Err(anyhow!("UQFF verification failed"))
    }
}

async fn run_inspect(
    model_id: String,
    quant: Option<String>,
    revision: Option<String>,
    token_source: &TokenSource,
) -> Result<()> {
    let resolved = resolve_uqff_artifacts(
        &model_id,
        revision.as_deref(),
        quant.as_deref(),
        token_source,
    )
    .await?;
    let inspection = inspect_uqff_artifacts(
        &resolved.artifacts,
        UqffReportOptions {
            generated_by: generated_by("mistralrs uqff inspect"),
            base_model: None,
            repo_id: None,
        },
    )
    .await
    .map_err(|e| anyhow!("{e}"))?;
    UqffExplorer::new(inspection).run()
}

struct ResolvedUqffArtifacts {
    artifacts: UqffArtifacts,
    local_write_path: Option<PathBuf>,
}

async fn resolve_uqff_artifacts(
    model_id: &str,
    revision: Option<&str>,
    quant: Option<&str>,
    token_source: &TokenSource,
) -> Result<ResolvedUqffArtifacts> {
    let revision = revision.unwrap_or(DEFAULT_REVISION);
    let files = list_model_files(model_id, revision, token_source, true).await?;
    let selected = select_uqff_files(&files, quant)?;
    let mut grouped: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for file in selected {
        grouped.entry(uqff_group_key(&file)).or_default().push(file);
    }

    let mut groups = Vec::with_capacity(grouped.len());
    for (quant, mut files) in grouped {
        files.sort_by_key(|file| uqff_shard_index(file).unwrap_or(0));
        let artifact_files = files
            .into_iter()
            .map(|file| {
                let model_id = model_id.to_string();
                let revision = revision.to_string();
                let token_source = token_source.clone();
                let name = file.clone();
                UqffArtifactFile::new(name, move |range: Range<u64>| {
                    let model_id = model_id.clone();
                    let revision = revision.clone();
                    let file = file.clone();
                    let token_source = token_source.clone();
                    async move {
                        read_model_file_range(&model_id, &revision, &file, range, &token_source)
                            .await
                            .map_err(|e| candle_core::Error::Msg(e.to_string()))
                    }
                })
            })
            .collect();
        groups.push(UqffArtifactGroup {
            quant,
            files: artifact_files,
        });
    }

    let selected_quants = groups
        .iter()
        .map(|group| group.quant.clone())
        .collect::<HashSet<_>>();
    let existing_report = read_existing_uqff_report(model_id, revision, &files, token_source)
        .await?
        .map(|mut report| {
            report
                .outputs
                .retain(|output| selected_quants.contains(&output.quant));
            report
        });
    let local_write_path = PathBuf::from(model_id)
        .exists()
        .then(|| PathBuf::from(model_id));
    Ok(ResolvedUqffArtifacts {
        artifacts: UqffArtifacts {
            groups,
            existing_report,
        },
        local_write_path,
    })
}

fn select_uqff_files(files: &[String], quant: Option<&str>) -> Result<Vec<String>> {
    let uqff_files = files
        .iter()
        .filter(|file| file.ends_with(".uqff"))
        .cloned()
        .collect::<Vec<_>>();
    if uqff_files.is_empty() {
        return Err(anyhow!("No `.uqff` files were found."));
    }
    let Some(raw_quant) = quant.map(str::trim).filter(|quant| !quant.is_empty()) else {
        return Ok(uqff_files);
    };
    if raw_quant.eq_ignore_ascii_case("all") {
        return Ok(uqff_files);
    }

    let first = resolve_uqff_shorthand(raw_quant, &uqff_files)
        .or_else(|| {
            uqff_files
                .iter()
                .find(|file| file.as_str() == raw_quant)
                .cloned()
        })
        .or_else(|| {
            let shard = format!("{raw_quant}-0.uqff");
            uqff_files
                .iter()
                .find(|file| file.as_str() == shard)
                .cloned()
        })
        .ok_or_else(|| {
            anyhow!(
                "No UQFF files matched `--quant {raw_quant}`. Available: {}",
                uqff_files.join(", ")
            )
        })?;
    let expanded = expand_uqff_shards(&first, &uqff_files);
    let expanded = if expanded.is_empty() {
        vec![first]
    } else {
        expanded
    };
    let mut seen = HashSet::new();
    Ok(expanded
        .into_iter()
        .filter(|file| seen.insert(file.clone()))
        .collect())
}

async fn read_existing_uqff_report(
    model_id: &str,
    revision: &str,
    files: &[String],
    token_source: &TokenSource,
) -> Result<Option<UqffReport>> {
    if !files.iter().any(|file| file == UQFF_REPORT_JSON) {
        return Ok(None);
    }
    let Some(path) = try_get_model_file(model_id, revision, UQFF_REPORT_JSON, token_source).await?
    else {
        return Ok(None);
    };
    let data = tokio::fs::read_to_string(&path).await?;
    serde_json::from_str(&data)
        .map(Some)
        .map_err(|e| anyhow!("{}: {e}", path.display()))
}

fn uqff_group_key(file: &str) -> String {
    let stem = file
        .rsplit_once('/')
        .map_or(file, |(_, name)| name)
        .strip_suffix(".uqff")
        .unwrap_or(file);
    if let Some((prefix, suffix)) = stem.rsplit_once('-') {
        if suffix.chars().all(|ch| ch.is_ascii_digit()) {
            return prefix.to_string();
        }
    }
    stem.to_string()
}

fn uqff_shard_index(file: &str) -> Option<u64> {
    file.rsplit_once('/')
        .map_or(file, |(_, name)| name)
        .strip_suffix(".uqff")
        .and_then(|stem| stem.rsplit_once('-'))
        .and_then(|(_, suffix)| suffix.parse().ok())
}

fn generated_by(tool: &str) -> UqffGeneratedBy {
    UqffGeneratedBy {
        tool: tool.to_string(),
        mistralrs_version: Some(mistralrs_core::MISTRALRS_VERSION.to_string()),
        git_revision: Some(mistralrs_core::MISTRALRS_GIT_REVISION.to_string()),
    }
}

fn print_report_summary(report: &UqffReport, verbose: bool) {
    println!("UQFF report");
    println!("  version: {}", report.uqff_version);
    if let Some(base_model) = &report.base_model {
        println!("  base model: {base_model}");
    }
    if let Some(repo_id) = &report.repo_id {
        println!("  repo id: {repo_id}");
    }
    println!("  generated by: {}", report.generated_by.tool);
    println!();
    println!("Outputs:");
    for output in &report.outputs {
        let counts = output
            .actual_counts
            .iter()
            .map(|(stored, count)| format!("{count} {stored}"))
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "  {}: {} layers, {} shard{}, {}, {} fallback{}",
            output.quant,
            output.layers,
            output.shards.len(),
            if output.shards.len() == 1 { "" } else { "s" },
            counts,
            output.fallback_count,
            if output.fallback_count == 1 { "" } else { "s" }
        );
        if verbose {
            for fallback in &output.fallbacks {
                let reason = fallback
                    .reason
                    .as_deref()
                    .map(|reason| format!(": {reason}"))
                    .unwrap_or_default();
                println!(
                    "    {}: {} -> {} {:?}{}",
                    fallback.module, fallback.from, fallback.to, fallback.shape, reason
                );
            }
        }
    }
}

#[derive(Clone)]
struct TensorInfo {
    group: String,
    name: String,
    display_name: Option<String>,
    dtype: String,
    shape: Vec<usize>,
    size_bytes: usize,
    labels: Vec<String>,
    stored: Option<QuantizedSerdeType>,
}

#[derive(Clone)]
struct MetadataInfo {
    name: String,
    value: String,
}

#[derive(Clone)]
enum TreeNode {
    Group {
        name: String,
        children: Vec<TreeNode>,
        expanded: bool,
        tensor_count: usize,
        total_size: usize,
        labels: Vec<String>,
    },
    Tensor(TensorInfo),
    Metadata(MetadataInfo),
}

impl TreeNode {
    fn name(&self) -> &str {
        match self {
            Self::Group { name, .. } => name,
            Self::Tensor(info) => &info.name,
            Self::Metadata(info) => &info.name,
        }
    }
}

struct UqffExplorer {
    title: String,
    summary: String,
    tensors: Vec<TensorInfo>,
    metadata: Vec<MetadataInfo>,
    tree: Vec<TreeNode>,
    flattened_tree: Vec<(TreeNode, usize)>,
    filtered_tree: Vec<(TreeNode, usize)>,
    selected_idx: usize,
    scroll_offset: usize,
    search_query: String,
    search_mode: bool,
}

impl UqffExplorer {
    fn new(inspection: UqffInspection) -> Self {
        let summary = inspection
            .report
            .outputs
            .iter()
            .map(|output| {
                format!(
                    "{}:{} layers/{} fallback{}",
                    output.quant,
                    output.layers,
                    output.fallback_count,
                    if output.fallback_count == 1 { "" } else { "s" }
                )
            })
            .collect::<Vec<_>>()
            .join(" | ");
        let title = inspection
            .report
            .base_model
            .clone()
            .unwrap_or_else(|| "UQFF artifact".to_string());
        let tensors = inspection
            .tensors
            .iter()
            .map(tensor_info_from_summary)
            .collect::<Vec<_>>();
        let metadata = inspection
            .metadata
            .iter()
            .map(metadata_info_from_summary)
            .collect::<Vec<_>>();
        let mut explorer = Self {
            title,
            summary,
            tensors,
            metadata,
            tree: Vec::new(),
            flattened_tree: Vec::new(),
            filtered_tree: Vec::new(),
            selected_idx: 0,
            scroll_offset: 0,
            search_query: String::new(),
            search_mode: false,
        };
        explorer.build_tree();
        explorer
    }

    fn run(&mut self) -> Result<()> {
        terminal::enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, terminal::Clear(ClearType::All), cursor::Hide)?;
        let result = self.interactive_loop();
        execute!(stdout, terminal::Clear(ClearType::All), cursor::Show)?;
        terminal::disable_raw_mode()?;
        result
    }

    fn interactive_loop(&mut self) -> Result<()> {
        loop {
            self.draw()?;
            if let Event::Key(key_event) = event::read()? {
                match key_event {
                    KeyEvent {
                        code: KeyCode::Char('q'),
                        ..
                    } => {
                        if self.search_mode {
                            self.exit_search_mode();
                        } else {
                            break;
                        }
                    }
                    KeyEvent {
                        code: KeyCode::Char('c'),
                        modifiers: KeyModifiers::CONTROL,
                        ..
                    } => break,
                    KeyEvent {
                        code: KeyCode::Char('/'),
                        ..
                    } => self.enter_search_mode(),
                    KeyEvent {
                        code: KeyCode::Esc, ..
                    } => self.exit_search_mode(),
                    KeyEvent {
                        code: KeyCode::Up, ..
                    } => self.move_selection(-1),
                    KeyEvent {
                        code: KeyCode::Down,
                        ..
                    } => self.move_selection(1),
                    KeyEvent {
                        code: KeyCode::Enter,
                        ..
                    }
                    | KeyEvent {
                        code: KeyCode::Char(' '),
                        ..
                    } => self.handle_selection()?,
                    KeyEvent {
                        code: KeyCode::Backspace,
                        ..
                    } if self.search_mode => {
                        self.search_query.pop();
                        self.update_filtered_tree();
                        self.selected_idx = 0;
                        self.scroll_offset = 0;
                    }
                    KeyEvent {
                        code: KeyCode::Char(c),
                        ..
                    } if self.search_mode => {
                        self.search_query.push(c);
                        self.update_filtered_tree();
                        self.selected_idx = 0;
                        self.scroll_offset = 0;
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    fn draw(&mut self) -> Result<()> {
        let tree = if self.search_mode {
            &self.filtered_tree
        } else {
            &self.flattened_tree
        };
        let mut stdout = io::stdout();
        execute!(
            stdout,
            terminal::Clear(ClearType::All),
            cursor::MoveTo(0, 0)
        )?;
        let (_, terminal_height) = terminal::size()?;
        let header_height = 4usize;
        let footer_height = 2usize;
        let available_height =
            (terminal_height as usize).saturating_sub(header_height + footer_height);

        writeln!(stdout, "UQFF Inspect - {}\r", self.title)?;
        writeln!(stdout, "{}\r", truncate(&self.summary, 120))?;
        if self.search_mode {
            writeln!(
                stdout,
                "SEARCH: {} | Enter/Esc exits search\r",
                if self.search_query.is_empty() {
                    "_"
                } else {
                    &self.search_query
                }
            )?;
        } else {
            writeln!(
                stdout,
                "Up/Down navigate, Enter/Space expand or detail, / search, q quit\r"
            )?;
        }
        writeln!(stdout, "{}\r", "=".repeat(80))?;

        let new_scroll_offset = if self.selected_idx >= self.scroll_offset + available_height {
            self.selected_idx.saturating_sub(available_height - 1)
        } else if self.selected_idx < self.scroll_offset {
            self.selected_idx
        } else {
            self.scroll_offset
        };

        for (actual_index, (node, depth)) in tree
            .iter()
            .enumerate()
            .skip(new_scroll_offset)
            .take(available_height)
        {
            let is_selected = actual_index == self.selected_idx;
            if is_selected {
                execute!(
                    stdout,
                    SetForegroundColor(Color::Black),
                    SetBackgroundColor(Color::White)
                )?;
            }
            draw_node(node, *depth, &mut stdout)?;
            if is_selected {
                execute!(stdout, ResetColor)?;
            }
        }

        execute!(stdout, cursor::MoveTo(0, terminal_height - 1))?;
        writeln!(
            stdout,
            "Selected: {}/{} | Matches: {}\r",
            self.selected_idx.saturating_add(1),
            tree.len(),
            tree.len()
        )?;
        stdout.flush()?;
        self.scroll_offset = new_scroll_offset;
        Ok(())
    }

    fn build_tree(&mut self) {
        let mut root = Vec::new();
        if !self.metadata.is_empty() {
            root.push(TreeNode::Group {
                name: "metadata".to_string(),
                children: self
                    .metadata
                    .iter()
                    .cloned()
                    .map(TreeNode::Metadata)
                    .collect(),
                expanded: false,
                tensor_count: 0,
                total_size: 0,
                labels: Vec::new(),
            });
        }
        root.extend(build_tensor_tree(&self.tensors));
        self.tree = root;
        self.flatten_tree();
    }

    fn flatten_tree(&mut self) {
        self.flattened_tree = flatten_tree(&self.tree);
        self.update_filtered_tree();
    }

    fn update_filtered_tree(&mut self) {
        if self.search_query.is_empty() {
            self.filtered_tree = self.flattened_tree.clone();
            return;
        }
        let matcher = SkimMatcherV2::default();
        let mut scored = Vec::new();
        for tensor in &self.tensors {
            let haystack = format!("{} {}", tensor.name, tensor.labels.join(" "));
            let haystack = if let Some(stored) = &tensor.stored {
                format!("{haystack} {}", stored.stored_label(&tensor.group))
            } else {
                haystack
            };
            if let Some(score) = matcher.fuzzy_match(&haystack, &self.search_query) {
                scored.push((TreeNode::Tensor(tensor.clone()), score));
            }
        }
        for metadata in &self.metadata {
            if let Some(score) = matcher.fuzzy_match(&metadata.name, &self.search_query) {
                scored.push((TreeNode::Metadata(metadata.clone()), score));
            }
        }
        scored.sort_by_key(|(_, score)| Reverse(*score));
        self.filtered_tree = scored.into_iter().map(|(node, _)| (node, 0)).collect();
    }

    fn move_selection(&mut self, delta: i32) {
        let tree = if self.search_mode {
            &self.filtered_tree
        } else {
            &self.flattened_tree
        };
        if tree.is_empty() {
            return;
        }
        self.selected_idx = if delta < 0 {
            self.selected_idx.saturating_sub((-delta) as usize)
        } else {
            (self.selected_idx + delta as usize).min(tree.len() - 1)
        };
    }

    fn enter_search_mode(&mut self) {
        if !self.search_mode {
            self.search_mode = true;
            self.search_query.clear();
            self.update_filtered_tree();
            self.selected_idx = 0;
            self.scroll_offset = 0;
        }
    }

    fn exit_search_mode(&mut self) {
        if self.search_mode {
            self.search_mode = false;
            self.search_query.clear();
            self.update_filtered_tree();
            self.selected_idx = 0;
            self.scroll_offset = 0;
        }
    }

    fn handle_selection(&mut self) -> Result<()> {
        if self.search_mode {
            self.exit_search_mode();
            return Ok(());
        }
        let Some((node, _)) = self.flattened_tree.get(self.selected_idx) else {
            return Ok(());
        };
        match node {
            TreeNode::Group { .. } => {
                let mut tree = self.tree.clone();
                let _ = toggle_node_by_index(self.selected_idx, &mut tree);
                self.tree = tree;
                self.flatten_tree();
            }
            TreeNode::Tensor(info) => draw_detail(&format_tensor_detail(info))?,
            TreeNode::Metadata(info) => draw_detail(&format_metadata_detail(info))?,
        }
        Ok(())
    }
}

fn tensor_info_from_summary(summary: &UqffTensorSummary) -> TensorInfo {
    TensorInfo {
        group: summary.group.clone(),
        name: format!("{}.{}", summary.group, summary.name),
        display_name: None,
        dtype: summary.dtype.clone(),
        shape: summary.shape.clone(),
        size_bytes: summary.size_bytes,
        labels: summary.labels.clone(),
        stored: summary.stored,
    }
}

fn metadata_info_from_summary(summary: &UqffMetadataSummary) -> MetadataInfo {
    MetadataInfo {
        name: format!("{}.{}", summary.shard, summary.key),
        value: summary.value.clone(),
    }
}

fn build_tensor_tree(tensors: &[TensorInfo]) -> Vec<TreeNode> {
    let mut root_map: HashMap<String, Vec<TensorInfo>> = HashMap::new();
    for tensor in tensors {
        let prefix = tensor
            .name
            .split('.')
            .next()
            .unwrap_or(&tensor.name)
            .to_string();
        root_map.entry(prefix).or_default().push(tensor.clone());
    }
    let mut tree = Vec::new();
    for (prefix, mut tensors) in root_map {
        tensors.sort_by_key(|tensor| natural_sort_key(&tensor.name));
        let tensor_count = tensors.len();
        let total_size = tensors.iter().map(|tensor| tensor.size_bytes).sum();
        let children = build_subtree(&tensors, &prefix);
        tree.push(TreeNode::Group {
            name: prefix,
            children,
            expanded: true,
            tensor_count,
            total_size,
            labels: Vec::new(),
        });
    }
    tree.sort_by_key(|node| natural_sort_key(node.name()));
    tree
}

fn build_subtree(tensors: &[TensorInfo], prefix: &str) -> Vec<TreeNode> {
    let mut groups: HashMap<String, Vec<TensorInfo>> = HashMap::new();
    let mut direct = Vec::new();
    let prefix_dot = format!("{prefix}.");
    let group_names = tensors
        .iter()
        .filter_map(|tensor| {
            let remaining = tensor.name.strip_prefix(&prefix_dot)?;
            remaining.split_once('.').map(|(name, _)| name.to_string())
        })
        .collect::<HashSet<_>>();
    for tensor in tensors {
        if tensor.name == prefix {
            direct.push(payload_tensor_info(tensor));
            continue;
        }
        let remaining = tensor
            .name
            .strip_prefix(&prefix_dot)
            .unwrap_or(&tensor.name);
        if let Some((next, _)) = remaining.split_once('.') {
            groups
                .entry(next.to_string())
                .or_default()
                .push(tensor.clone());
        } else if group_names.contains(remaining) {
            groups
                .entry(remaining.to_string())
                .or_default()
                .push(tensor.clone());
        } else {
            direct.push(tensor.clone());
        }
    }
    let mut nodes = direct.into_iter().map(TreeNode::Tensor).collect::<Vec<_>>();
    for (name, group_tensors) in groups {
        let tensor_count = group_tensors.len();
        let total_size = group_tensors.iter().map(|tensor| tensor.size_bytes).sum();
        let full_prefix = format!("{prefix}.{name}");
        let labels = group_labels(&group_tensors, &name);
        nodes.push(TreeNode::Group {
            name,
            children: build_subtree(&group_tensors, &full_prefix),
            expanded: false,
            tensor_count,
            total_size,
            labels,
        });
    }
    nodes.sort_by_key(|node| natural_sort_key(node.name()));
    nodes
}

fn payload_tensor_info(tensor: &TensorInfo) -> TensorInfo {
    let mut tensor = tensor.clone();
    tensor.display_name = Some("data".to_string());
    tensor
}

fn group_labels(tensors: &[TensorInfo], name: &str) -> Vec<String> {
    if name != "weight" {
        return Vec::new();
    }
    let mut labels = Vec::new();
    let mut stored = tensors
        .iter()
        .filter_map(|tensor| {
            tensor
                .stored
                .map(|stored| stored.stored_label(&tensor.group))
        })
        .collect::<Vec<_>>();
    stored.sort_unstable();
    stored.dedup();
    match stored.as_slice() {
        [] => {}
        [stored] => push_label(&mut labels, stored),
        _ => push_label(&mut labels, "mixed"),
    }
    for label in tensors.iter().flat_map(|tensor| tensor.labels.iter()) {
        push_label(&mut labels, label);
    }
    labels
}

fn push_label(labels: &mut Vec<String>, label: &str) {
    if !labels.iter().any(|existing| existing == label) {
        labels.push(label.to_string());
    }
}

fn format_labels(labels: &[String]) -> String {
    if labels.is_empty() {
        String::new()
    } else {
        format!(" [{}]", labels.join(","))
    }
}

fn flatten_tree(tree: &[TreeNode]) -> Vec<(TreeNode, usize)> {
    let mut flattened = Vec::new();
    for node in tree {
        flatten_node(node, 0, &mut flattened);
    }
    flattened
}

fn flatten_node(node: &TreeNode, depth: usize, flattened: &mut Vec<(TreeNode, usize)>) {
    flattened.push((node.clone(), depth));
    if let TreeNode::Group {
        children, expanded, ..
    } = node
    {
        if *expanded {
            for child in children {
                flatten_node(child, depth + 1, flattened);
            }
        }
    }
}

fn toggle_node_by_index(target_idx: usize, nodes: &mut [TreeNode]) -> bool {
    let mut current_idx = 0;
    toggle_node_by_index_inner(target_idx, nodes, &mut current_idx)
}

fn toggle_node_by_index_inner(
    target_idx: usize,
    nodes: &mut [TreeNode],
    current_idx: &mut usize,
) -> bool {
    for node in nodes {
        if *current_idx == target_idx {
            if let TreeNode::Group { expanded, .. } = node {
                *expanded = !*expanded;
                return true;
            }
            return false;
        }
        *current_idx += 1;
        if let TreeNode::Group {
            children, expanded, ..
        } = node
        {
            if *expanded && toggle_node_by_index_inner(target_idx, children, current_idx) {
                return true;
            }
        }
    }
    false
}

fn draw_node(node: &TreeNode, depth: usize, stdout: &mut io::Stdout) -> Result<()> {
    let indent = "  ".repeat(depth);
    match node {
        TreeNode::Group {
            name,
            expanded,
            tensor_count,
            total_size,
            labels,
            ..
        } => {
            let icon = if *expanded { "v" } else { ">" };
            let labels = format_labels(labels);
            writeln!(
                stdout,
                "{}{} {}{} ({} tensors, {})\r",
                indent,
                icon,
                name,
                labels,
                tensor_count,
                format_size(*total_size)
            )?;
        }
        TreeNode::Tensor(info) => {
            let display_name = info
                .display_name
                .as_deref()
                .unwrap_or_else(|| info.name.split('.').next_back().unwrap_or(&info.name));
            let labels = format_labels(&info.labels);
            writeln!(
                stdout,
                "{}  {} [{}, {}, {}]{}\r",
                indent,
                display_name,
                info.dtype,
                format_shape(&info.shape),
                format_size(info.size_bytes),
                labels
            )?;
        }
        TreeNode::Metadata(info) => {
            writeln!(
                stdout,
                "{}  {}: {}\r",
                indent,
                info.name,
                truncate(&info.value, 64)
            )?;
        }
    }
    Ok(())
}

fn draw_detail(lines: &[String]) -> Result<()> {
    let mut stdout = io::stdout();
    execute!(
        stdout,
        terminal::Clear(ClearType::All),
        cursor::MoveTo(0, 0)
    )?;
    for line in lines.iter().take(30) {
        writeln!(stdout, "{line}\r")?;
    }
    writeln!(stdout, "\rPress any key to return...\r")?;
    stdout.flush()?;
    let _ = event::read();
    Ok(())
}

fn format_tensor_detail(info: &TensorInfo) -> Vec<String> {
    vec![
        "Tensor details".to_string(),
        "==============".to_string(),
        format!("Name: {}", info.name),
        format!("DType: {}", info.dtype),
        format!("Shape: {}", format_shape(&info.shape)),
        format!("Size: {}", format_size(info.size_bytes)),
        format!(
            "Stored: {}",
            info.stored
                .map(|stored| stored.stored_label(&info.group))
                .unwrap_or_else(|| "unknown".to_string())
        ),
        format!("Labels: {}", info.labels.join(", ")),
    ]
}

fn format_metadata_detail(info: &MetadataInfo) -> Vec<String> {
    let mut lines = vec![
        "Metadata details".to_string(),
        "================".to_string(),
        format!("Key: {}", info.name),
        "Value:".to_string(),
    ];
    lines.extend(info.value.lines().map(|line| format!("  {line}")));
    lines
}

fn format_shape(shape: &[usize]) -> String {
    format!(
        "({})",
        shape
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn format_size(bytes: usize) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;
    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }
    if unit_idx == 0 {
        format!("{} {}", bytes, UNITS[unit_idx])
    } else {
        format!("{:.1} {}", size, UNITS[unit_idx])
    }
}

fn truncate(value: &str, max: usize) -> String {
    if value.len() <= max {
        value.to_string()
    } else {
        format!("{}...", &value[..max.saturating_sub(3)])
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum NaturalSortItem {
    Text(String),
    Number(u32),
}

fn natural_sort_key(name: &str) -> Vec<NaturalSortItem> {
    let mut result = Vec::new();
    let mut current_number = String::new();
    let mut current_text = String::new();
    for ch in name.chars() {
        if ch.is_ascii_digit() {
            if !current_text.is_empty() {
                result.push(NaturalSortItem::Text(std::mem::take(&mut current_text)));
            }
            current_number.push(ch);
        } else {
            if !current_number.is_empty() {
                if let Ok(num) = current_number.parse::<u32>() {
                    result.push(NaturalSortItem::Number(num));
                } else {
                    result.push(NaturalSortItem::Text(std::mem::take(&mut current_number)));
                }
                current_number.clear();
            }
            current_text.push(ch);
        }
    }
    if !current_number.is_empty() {
        if let Ok(num) = current_number.parse::<u32>() {
            result.push(NaturalSortItem::Number(num));
        } else {
            result.push(NaturalSortItem::Text(current_number));
        }
    }
    if !current_text.is_empty() {
        result.push(NaturalSortItem::Text(current_text));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(name: &str, size_bytes: usize) -> TensorInfo {
        TensorInfo {
            group: "afq2".to_string(),
            name: name.to_string(),
            display_name: None,
            dtype: "U8".to_string(),
            shape: Vec::new(),
            size_bytes,
            labels: Vec::new(),
            stored: None,
        }
    }

    fn child_group<'a>(nodes: &'a [TreeNode], name: &str) -> &'a [TreeNode] {
        nodes
            .iter()
            .find_map(|node| match node {
                TreeNode::Group {
                    name: group_name,
                    children,
                    ..
                } if group_name == name => Some(children.as_slice()),
                _ => None,
            })
            .unwrap_or_else(|| panic!("missing group `{name}`"))
    }

    #[test]
    fn tensor_prefix_collision_is_nested_as_group_payload() {
        let tree = build_tensor_tree(&[
            tensor("afq2.model.embed_audio.embedding_projection.weight", 600),
            tensor("afq2.model.embed_audio.embedding_projection.weight.bits", 1),
        ]);

        let root = child_group(&tree, "afq2");
        let model = child_group(root, "model");
        let embed_audio = child_group(model, "embed_audio");
        let projection = child_group(embed_audio, "embedding_projection");

        assert!(!projection.iter().any(|node| matches!(
            node,
            TreeNode::Tensor(info) if info.name.ends_with(".weight")
        )));

        let weight = child_group(projection, "weight");
        assert!(weight.iter().any(|node| matches!(
            node,
            TreeNode::Tensor(info)
                if info.name.ends_with(".weight") && info.display_name.as_deref() == Some("data")
        )));
        assert!(weight.iter().any(|node| matches!(
            node,
            TreeNode::Tensor(info) if info.name.ends_with(".weight.bits")
        )));
    }
}
