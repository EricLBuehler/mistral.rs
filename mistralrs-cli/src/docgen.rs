//! Generates the Starlight CLI reference pages from the clap tree.
//! `cli_reference_matches_committed` is the golden test keeping the committed pages in sync;
//! refresh them with `cargo test -p mistralrs-cli regenerate_cli_reference -- --ignored`.

use std::collections::HashSet;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use clap::{Arg, Command, CommandFactory};

use crate::args::Cli;

const COMMITTED_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../docs/src/content/docs/reference/cli"
);
const OUT_DIR_ENV: &str = "CLI_DOCS_OUT";
const LINK_BASE: &str = "/mistral.rs/reference/cli";
const REGEN_HINT: &str = "cargo test -p mistralrs-cli regenerate_cli_reference -- --ignored";
const GENERATED_NOTICE: &str =
    "<!-- Generated from clap definitions by mistralrs-cli docgen. Do not edit. -->\n\n";
const SKIPPED_ARG_IDS: &[&str] = &["help", "version"];

fn yaml_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn cell(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', " ").trim().to_string()
}

fn about_line(cmd: &Command) -> String {
    cmd.get_about()
        .map(|s| s.to_string())
        .unwrap_or_default()
        .lines()
        .next()
        .unwrap_or_default()
        .to_string()
}

fn flag_syntax(arg: &Arg) -> String {
    let mut parts = Vec::new();
    if let Some(short) = arg.get_short() {
        parts.push(format!("-{short}"));
    }
    if let Some(long) = arg.get_long() {
        parts.push(format!("--{long}"));
    }
    let mut out = parts.join(", ");
    let value_names: Vec<String> = arg
        .get_value_names()
        .map(|names| names.iter().map(|n| n.to_string()).collect())
        .unwrap_or_else(|| vec![arg.get_id().to_string().to_uppercase()]);
    if out.is_empty() {
        return format!("<{}>", value_names.join("> <"));
    }
    if arg.get_action().takes_values() {
        write!(out, " <{}>", value_names.join("> <")).unwrap();
    }
    out
}

fn arg_description(arg: &Arg) -> String {
    let mut desc = arg
        .get_long_help()
        .or_else(|| arg.get_help())
        .map(|s| s.to_string())
        .unwrap_or_default();
    let possible: Vec<String> = arg
        .get_possible_values()
        .iter()
        .filter(|pv| !pv.is_hide_set())
        .map(|pv| format!("`{}`", pv.get_name()))
        .collect();
    if !possible.is_empty() {
        if !desc.is_empty() {
            desc.push(' ');
        }
        write!(desc, "Possible values: {}.", possible.join(", ")).unwrap();
    }
    desc
}

fn arg_default(arg: &Arg) -> String {
    let defaults: Vec<String> = arg
        .get_default_values()
        .iter()
        .map(|v| format!("`{}`", v.to_string_lossy()))
        .collect();
    if !defaults.is_empty() {
        defaults.join(", ")
    } else if arg.is_required_set() {
        "required".to_string()
    } else {
        String::new()
    }
}

fn write_table<'a>(md: &mut String, args: impl Iterator<Item = &'a Arg>) {
    let args: Vec<&Arg> = args
        .filter(|a| !a.is_hide_set() && !SKIPPED_ARG_IDS.contains(&a.get_id().as_str()))
        .collect();
    if args.is_empty() {
        return;
    }
    md.push_str("| Option | Default | Description |\n|---|---|---|\n");
    for arg in args {
        writeln!(
            md,
            "| `{}` | {} | {} |",
            cell(&flag_syntax(arg)),
            cell(&arg_default(arg)),
            cell(&arg_description(arg)),
        )
        .unwrap();
    }
    md.push('\n');
}

fn usage_block(cmd: &Command, path: &str) -> String {
    cmd.clone()
        .render_usage()
        .to_string()
        .lines()
        .map(|line| {
            let line = line.trim().trim_start_matches("Usage: ").trim_start();
            match line.strip_prefix(cmd.get_name()) {
                Some(rest) => format!("{path}{rest}"),
                None => line.to_string(),
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn visible_subcommands(cmd: &Command) -> impl Iterator<Item = &Command> {
    cmd.get_subcommands()
        .filter(|s| !s.is_hide_set() && s.get_name() != "help")
}

fn write_command_section(md: &mut String, cmd: &Command, path: &str, depth: usize) {
    if depth > 0 {
        writeln!(md, "{} {}\n", "#".repeat((depth + 1).min(6)), path).unwrap();
    }
    let about = cmd
        .get_long_about()
        .or_else(|| cmd.get_about())
        .map(|s| s.to_string())
        .unwrap_or_default();
    if !about.is_empty() {
        writeln!(md, "{about}\n").unwrap();
    }
    writeln!(md, "```\n{}\n```\n", usage_block(cmd, path)).unwrap();
    write_table(md, cmd.get_arguments().filter(|a| !a.is_global_set()));
    for sub in visible_subcommands(cmd) {
        write_command_section(md, sub, &format!("{path} {}", sub.get_name()), depth + 1);
    }
}

fn render_page(cmd: &Command, path: &str, order: usize) -> String {
    let mut md = String::new();
    writeln!(
        md,
        "---\ntitle: \"{path}\"\ndescription: \"{}\"\nsidebar:\n  order: {order}\n---\n",
        yaml_escape(&about_line(cmd)),
    )
    .unwrap();
    md.push_str(GENERATED_NOTICE);
    write_command_section(&mut md, cmd, path, 0);
    md
}

fn render_index(root: &Command) -> String {
    let mut md = String::new();
    md.push_str("---\ntitle: \"CLI reference\"\ndescription: \"Subcommands and flags of the mistralrs binary.\"\nsidebar:\n  order: 1\n---\n\n");
    md.push_str(GENERATED_NOTICE);
    md.push_str("## Subcommands\n\n| Subcommand | Purpose |\n|---|---|\n");
    for sub in visible_subcommands(root) {
        writeln!(
            md,
            "| [`mistralrs {0}`]({LINK_BASE}/{0}/) | {1} |",
            sub.get_name(),
            cell(&about_line(sub)),
        )
        .unwrap();
    }
    md.push_str("\n## Global options\n\n");
    write_table(&mut md, root.get_arguments());
    md
}

fn rendered_pages() -> Vec<(String, String)> {
    let mut root = Cli::command();
    root.build();
    let mut pages = vec![("index.md".to_string(), render_index(&root))];
    for (i, sub) in visible_subcommands(&root).enumerate() {
        let path = format!("mistralrs {}", sub.get_name());
        pages.push((
            format!("{}.md", sub.get_name()),
            render_page(sub, &path, i + 2),
        ));
    }
    pages
}

#[test]
fn cli_reference_matches_committed() {
    let dir = Path::new(COMMITTED_DIR);
    let pages = rendered_pages();
    let mut mismatches = Vec::new();
    for (name, content) in &pages {
        match fs::read_to_string(dir.join(name)) {
            Ok(on_disk) if &on_disk == content => {}
            Ok(_) => mismatches.push(format!("{name}: content differs")),
            Err(_) => mismatches.push(format!("{name}: missing")),
        }
    }
    let expected: HashSet<&str> = pages.iter().map(|(n, _)| n.as_str()).collect();
    for entry in fs::read_dir(dir).expect("committed CLI reference dir exists") {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        if name.ends_with(".md") && !expected.contains(name.as_str()) {
            mismatches.push(format!("{name}: stale (no longer generated)"));
        }
    }
    assert!(
        mismatches.is_empty(),
        "generated CLI reference is out of date:\n  {}\nregenerate with: {REGEN_HINT}",
        mismatches.join("\n  "),
    );
}

#[test]
#[ignore = "writes the committed CLI reference pages"]
fn regenerate_cli_reference() {
    let out_dir = std::env::var(OUT_DIR_ENV).unwrap_or_else(|_| COMMITTED_DIR.to_string());
    let out_dir = Path::new(&out_dir);
    fs::create_dir_all(out_dir).unwrap();
    let pages = rendered_pages();
    let expected: HashSet<&str> = pages.iter().map(|(n, _)| n.as_str()).collect();
    for entry in fs::read_dir(out_dir).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".md") && !expected.contains(name.as_str()) {
            fs::remove_file(entry.path()).unwrap();
        }
    }
    for (name, content) in pages {
        fs::write(out_dir.join(name), content).unwrap();
    }
}
