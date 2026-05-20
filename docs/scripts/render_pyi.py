#!/usr/bin/env python3
"""
Render mistralrs-pyo3/mistralrs.pyi as Starlight Markdown pages.

The .pyi file is the single source of truth for the Python API. This script
parses it with `ast` and writes one Markdown file per logical group into
`docs/src/content/docs/reference/python/`. The Starlight sidebar picks
them up via its `autogenerate` rule.

Run from the repo root or the docs directory.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from textwrap import dedent

SCRIPT_DIR = Path(__file__).resolve().parent
WEBSITE_DIR = SCRIPT_DIR.parent
REPO_DIR = WEBSITE_DIR.parent
PYI_PATH = REPO_DIR / "mistralrs-pyo3" / "mistralrs.pyi"
OUT_DIR = WEBSITE_DIR / "src" / "content" / "docs" / "reference" / "python"
STUB_REL = "mistralrs-pyo3/mistralrs.pyi"

# (title, slug, description, [class names to include])
GROUPS = [
    (
        "Runner",
        "runner",
        "The main entry point. Load a model and send requests.",
        ["Runner"],
    ),
    (
        "Which",
        "which",
        "Variants that select which kind of model to load.",
        ["Which"],
    ),
    (
        "Requests",
        "requests",
        "Request dataclasses passed to Runner methods.",
        ["ChatCompletionRequest", "CompletionRequest", "EmbeddingRequest"],
    ),
    (
        "Responses",
        "responses",
        "Response and streaming types returned by the engine.",
        [
            "ChatCompletionResponse",
            "ChatCompletionChunkResponse",
            "AgenticToolCallRecord",
            "Choice",
            "ChunkChoice",
            "Delta",
            "ResponseMessage",
            "CompletionResponse",
            "CompletionChoice",
            "Usage",
            "Logprobs",
            "ResponseLogprob",
            "TopLogprob",
            "ImageGenerationResponse",
            "ImageChoice",
            "SpeechGenerationResponse",
            "ToolCallResponse",
            "ToolCallType",
            "CalledFunction",
        ],
    ),
    (
        "Enums",
        "enums",
        "Architecture, dtype, and option enums.",
        [
            "Architecture",
            "EmbeddingArchitecture",
            "MultimodalArchitecture",
            "DiffusionArchitecture",
            "SpeechLoaderType",
            "ModelDType",
            "IsqOrganization",
            "ImageGenerationResponseFormat",
            "ToolChoice",
            "SearchContextSize",
            "AgentPermission",
            "CodeExecutionPermission",
            "NetworkMode",
            "AgentToolSource",
            "AgentToolKind",
            "AgentToolApprovalDecisionKind",
            "PagedCacheType",
        ],
    ),
    (
        "Search",
        "search",
        "Types for web-search tool configuration.",
        ["WebSearchOptions", "WebSearchUserLocation", "ApproximateUserLocation"],
    ),
    (
        "AnyMoE",
        "anymoe",
        "AnyMoE expert and config types.",
        ["AnyMoeExpertType", "AnyMoeConfig"],
    ),
    (
        "Code execution",
        "code-execution",
        "Configuration for the built-in Python code executor.",
        ["SandboxPolicy", "CodeExecutionConfig"],
    ),
    (
        "Agent approvals",
        "agent-approvals",
        "Request and decision types for agent action approval callbacks.",
        ["AgentToolMetadata", "AgentToolApproval", "AgentToolApprovalDecision"],
    ),
    (
        "Files",
        "files",
        "First-class output files surfaced from agentic runs.",
        ["RequestedFile", "FileSource", "File"],
    ),
    (
        "MCP",
        "mcp",
        "MCP client configuration types.",
        ["McpServerSourcePy", "McpServerConfigPy", "McpClientConfigPy"],
    ),
    (
        "Auto-mapping",
        "automap",
        "Hints for automatic device mapping.",
        ["TextAutoMapParams", "MultimodalAutoMapParams"],
    ),
]

SIG_WRAP_THRESHOLD = 70  # wrap arg list onto multiple lines beyond this width


def _unparse(node) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _is_enum(cls: ast.ClassDef) -> bool:
    return any(isinstance(b, ast.Name) and b.id == "Enum" for b in cls.bases)


def _collect_args(func: ast.FunctionDef) -> list[tuple[str, str, str | None]]:
    """Return [(name, annotation, default_str_or_None)] skipping `self`."""
    out: list[tuple[str, str, str | None]] = []
    args = func.args
    defaults = list(args.defaults)
    positional = list(args.args)
    pad = len(positional) - len(defaults)
    for i, arg in enumerate(positional):
        if arg.arg == "self":
            continue
        ann = _unparse(arg.annotation)
        default_idx = i - pad
        default = _unparse(defaults[default_idx]) if default_idx >= 0 else None
        out.append((arg.arg, ann, default))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        ann = _unparse(arg.annotation)
        d = _unparse(default) if default is not None else None
        out.append((arg.arg, ann, d))
    return out


def _format_signature_block(func_name: str, func: ast.FunctionDef) -> str:
    """Format a signature as a Python code block, wrapping onto multiple lines
    when the single-line form exceeds SIG_WRAP_THRESHOLD."""
    ret = _unparse(func.returns)
    args = _collect_args(func)

    def fmt(a: tuple[str, str, str | None]) -> str:
        name, ann, default = a
        s = name
        if ann:
            s += f": {ann}"
        if default is not None:
            s += f" = {default}"
        return s

    parts = [fmt(a) for a in args]
    single = f"{func_name}({', '.join(parts)})"
    if ret:
        single += f" -> {ret}"

    if len(single) <= SIG_WRAP_THRESHOLD:
        return single

    # Multi-line form with four-space indent for each arg.
    indent = "    "
    multi = [f"{func_name}("]
    for p in parts:
        multi.append(f"{indent}{p},")
    closing = ")"
    if ret:
        closing += f" -> {ret}"
    multi.append(closing)
    return "\n".join(multi)


# Matches a docstring "Args:" block and captures indented argument descriptions.
ARGS_SECTION_RE = re.compile(r"(?m)^[ \t]*Args?\s*:\s*\n(?P<body>(?:[ \t]+[^\n]*\n?)+)")
RETURNS_SECTION_RE = re.compile(
    r"(?m)^[ \t]*Returns?\s*:\s*\n(?P<body>(?:[ \t]+[^\n]*\n?)+)"
)
RAISES_SECTION_RE = re.compile(
    r"(?m)^[ \t]*Raises\s*:\s*\n(?P<body>(?:[ \t]+[^\n]*\n?)+)"
)


def _parse_doc_sections(doc: str) -> tuple[str, list[tuple[str, str]], str, str]:
    """Pull Args/Returns/Raises out of a docstring.

    Returns (summary, params, returns_text, raises_text).
    `summary` is everything before the first recognized section.
    `params` is [(name, description)] extracted from Args:.
    """
    if not doc:
        return "", [], "", ""

    # Find earliest section start.
    sections = []
    for pat in (ARGS_SECTION_RE, RETURNS_SECTION_RE, RAISES_SECTION_RE):
        m = pat.search(doc)
        if m:
            sections.append(m.start())
    cut = min(sections) if sections else len(doc)
    summary = doc[:cut].strip()

    params: list[tuple[str, str]] = []
    m = ARGS_SECTION_RE.search(doc)
    if m:
        body = dedent(m.group("body")).strip("\n")
        # Each entry: `name: description` possibly multi-line (continuation
        # indented deeper). We handle simple `name: desc` per line.
        current_name: str | None = None
        current_desc: list[str] = []
        for line in body.splitlines():
            stripped = line.rstrip()
            if not stripped:
                continue
            # top-level entry: starts at col 0 after dedent, matches `name: ...`
            lead_ws = len(line) - len(line.lstrip(" "))
            if lead_ws == 0 and ":" in stripped:
                # flush previous
                if current_name is not None:
                    params.append((current_name, " ".join(current_desc).strip()))
                name, _, rest = stripped.partition(":")
                current_name = name.strip()
                current_desc = [rest.strip()]
            else:
                current_desc.append(stripped.strip())
        if current_name is not None:
            params.append((current_name, " ".join(current_desc).strip()))

    returns_text = ""
    m = RETURNS_SECTION_RE.search(doc)
    if m:
        returns_text = dedent(m.group("body")).strip()

    raises_text = ""
    m = RAISES_SECTION_RE.search(doc)
    if m:
        raises_text = dedent(m.group("body")).strip()

    return summary, params, returns_text, raises_text


def _clean_doc(doc: str | None) -> str:
    if not doc:
        return ""
    return dedent(doc).strip()


def _md_escape_cell(text: str) -> str:
    """Escape a string for safe inclusion in a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")


def _md_code_cell(text: str) -> str:
    """Inline-code a value and escape pipes so it survives a table cell."""
    if not text:
        return ""
    # Pipes inside backticks still end the cell in CommonMark tables; escape them.
    return "`" + text.replace("|", "\\|") + "`"


def _render_params_table(
    args: list[tuple[str, str, str | None]],
    param_docs: dict[str, str],
) -> str:
    lines = [
        "**Parameters**",
        "",
        "| Name | Type | Default | Description |",
        "| --- | --- | --- | --- |",
    ]
    for name, ann, default in args:
        type_cell = _md_code_cell(ann)
        default_cell = _md_code_cell(default) if default is not None else "required"
        desc = _md_escape_cell(param_docs.get(name, ""))
        lines.append(f"| `{name}` | {type_cell} | {default_cell} | {desc} |")
    lines.append("")
    return "\n".join(lines)


def _render_function(
    func: ast.FunctionDef, heading: str, owner: str | None = None
) -> str:
    display_name = "__init__" if func.name == "__init__" else func.name
    anchor = f"{owner}.{display_name}" if owner else display_name

    sig_block = _format_signature_block(func.name, func)
    doc = _clean_doc(ast.get_docstring(func))
    summary, param_docs_list, returns_text, raises_text = _parse_doc_sections(doc)
    param_docs = {name: desc for name, desc in param_docs_list}

    args = _collect_args(func)
    lines = [f"{heading} `{anchor}`", ""]
    lines.append("```text")
    lines.append(sig_block)
    lines.append("```")
    lines.append("")

    # Prefer explicit Args:/Returns:, fall back to free-form summary.
    if summary:
        lines.append(summary)
        lines.append("")

    if args:
        # If we have Args: docs, use a table; otherwise emit a plain "Parameters" list
        # only when there is anything useful to show beyond the signature.
        if param_docs:
            lines.append(_render_params_table(args, param_docs))

    if returns_text:
        lines.append(f"**Returns:** {returns_text}")
        lines.append("")

    if raises_text:
        lines.append(f"**Raises:** {raises_text}")
        lines.append("")

    return "\n".join(lines)


def _render_fields_table(owner: str, fields: list[tuple[str, str, str]]) -> str:
    has_default = any(v for _, _, v in fields)
    if has_default:
        lines = [
            "| Field | Type | Default |",
            "| --- | --- | --- |",
        ]
        for name, ftype, value in fields:
            t = _md_code_cell(ftype)
            d = _md_code_cell(value) if value else "required"
            lines.append(f"| `{name}` | {t} | {d} |")
    else:
        lines = [
            "| Field | Type |",
            "| --- | --- |",
        ]
        for name, ftype, _ in fields:
            t = _md_code_cell(ftype)
            lines.append(f"| `{name}` | {t} |")
    lines.append("")
    return "\n".join(lines)


def _render_enum_table(owner: str, values: list[tuple[str, str]]) -> str:
    has_value = any(v for _, v in values)
    if has_value:
        lines = [
            "Members and their wire/config names where relevant. The members are fieldless PyO3 enum variants and do not expose `.value`.",
            "",
            "| Member | Wire/config name |",
            "| --- | --- |",
        ]
        for name, value in values:
            v = _md_code_cell(value)
            lines.append(f"| `{owner}.{name}` | {v} |")
    else:
        lines = [
            "| Member |",
            "| --- |",
        ]
        for name, _ in values:
            lines.append(f"| `{owner}.{name}` |")
    lines.append("")
    return "\n".join(lines)


def _render_class(
    cls: ast.ClassDef, heading: str = "###", parent: str | None = None
) -> str:
    is_enum = _is_enum(cls)
    full_name = f"{parent}.{cls.name}" if parent else cls.name
    lines: list[str] = [f"{heading} `{full_name}`", ""]
    doc = _clean_doc(ast.get_docstring(cls))
    if doc:
        lines.append(doc)
        lines.append("")

    fields: list[tuple[str, str, str]] = []
    enum_values: list[tuple[str, str]] = []
    methods: list[ast.FunctionDef] = []
    nested: list[ast.ClassDef] = []

    for item in cls.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            name = item.target.id
            ftype = _unparse(item.annotation)
            value = _unparse(item.value) if item.value is not None else ""
            fields.append((name, ftype, value))
        elif isinstance(item, ast.Assign):
            for t in item.targets:
                if isinstance(t, ast.Name):
                    enum_values.append((t.id, _unparse(item.value)))
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.append(item)
        elif isinstance(item, ast.ClassDef):
            nested.append(item)

    if is_enum:
        values = enum_values + [(n, v) for n, _, v in fields if v]
        if values:
            lines.append(_render_enum_table(cls.name, values))
        for nc in nested:
            sub_heading = "#" * (len(heading) + 1)
            lines.append(_render_class(nc, sub_heading, parent=cls.name))
    else:
        if fields:
            lines.append(_render_fields_table(full_name, fields))
        for nc in nested:
            sub_heading = "#" * (len(heading) + 1)
            lines.append(_render_class(nc, sub_heading, parent=full_name))

    # __init__ first if present, then other methods
    init = [m for m in methods if m.name == "__init__"]
    rest = [m for m in methods if m.name != "__init__"]
    sub_heading = "#" * (len(heading) + 1)
    for m in init + rest:
        lines.append(_render_function(m, sub_heading, owner=full_name))

    return "\n".join(lines)


def _render_page(
    title: str,
    description: str,
    class_names: list[str],
    classes_by_name: dict[str, ast.ClassDef],
    order: int,
) -> str:
    # YAML-safe quoting.
    safe_description = description.replace('"', '\\"')
    frontmatter = [
        "---",
        f"title: {title}",
        f'description: "{safe_description}"',
        "sidebar:",
        f"  order: {order}",
        "---",
        "",
    ]
    body: list[str] = []
    for name in class_names:
        cls = classes_by_name.get(name)
        if cls is None:
            print(f"warning: class {name!r} not in .pyi", file=sys.stderr)
            continue
        body.append(_render_class(cls, heading="##"))
        body.append("")

    footer = [
        "---",
        "",
        f"<small>Generated from [`{STUB_REL}`](https://github.com/EricLBuehler/mistral.rs/blob/master/{STUB_REL}).</small>",
        "",
    ]

    return "\n".join(frontmatter) + "\n".join(body) + "\n".join(footer)


def _render_index() -> str:
    lines = [
        "---",
        "title: Python API",
        'description: "The mistralrs Python package."',
        "sidebar:",
        "  order: 6",
        "---",
        "",
        "The `mistralrs` Python package exposes the same engine that powers the `mistralrs` CLI.",
        "",
        "## Install",
        "",
        "One wheel per accelerator. All wheels expose the same `mistralrs` module.",
        "",
        "| Accelerator | Package |",
        "| --- | --- |",
        "| CPU (or Intel CPU with MKL) | `pip install mistralrs` |",
        "| NVIDIA GPU | `pip install mistralrs-cuda` |",
        "| Apple Silicon | `pip install mistralrs-metal` |",
        "| Intel MKL (pinned) | `pip install mistralrs-mkl` |",
        "| macOS Accelerate | `pip install mistralrs-accelerate` |",
        "",
        "## Pages",
        "",
        "| Page | Covers |",
        "| --- | --- |",
    ]
    for title, slug, desc, _ in GROUPS:
        lines.append(f"| [{title}](/mistral.rs/reference/python/{slug}/) | {desc} |")
    lines.append("")
    lines.append(
        "See [Tutorial 3](/mistral.rs/tutorials/03-python-sdk/) for a walkthrough and the [Python guides](/mistral.rs/guides/python/) for task-oriented recipes."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        f"<small>Generated from [`{STUB_REL}`](https://github.com/EricLBuehler/mistral.rs/blob/master/{STUB_REL}).</small>"
    )
    lines.append("")
    return "\n".join(lines)


def _collect_classes(tree: ast.Module) -> dict[str, ast.ClassDef]:
    out: dict[str, ast.ClassDef] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            out[node.name] = node
    return out


def main() -> int:
    if not PYI_PATH.exists():
        print(f"error: {PYI_PATH} does not exist", file=sys.stderr)
        return 1
    source = PYI_PATH.read_text()
    tree = ast.parse(source)
    classes_by_name = _collect_classes(tree)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Clean previous output so removed classes do not linger.
    for existing in OUT_DIR.glob("*.md"):
        existing.unlink()

    (OUT_DIR / "index.md").write_text(_render_index())
    print(f"wrote {OUT_DIR / 'index.md'}")

    for i, (title, slug, desc, names) in enumerate(GROUPS, start=2):
        page = _render_page(title, desc, names, classes_by_name, i)
        path = OUT_DIR / f"{slug}.md"
        path.write_text(page)
        print(f"wrote {path}")

    documented = {n for _, _, _, names in GROUPS for n in names}
    uncovered = sorted(set(classes_by_name) - documented)
    if uncovered:
        print(
            f"note: {len(uncovered)} .pyi classes not covered by any group: "
            + ", ".join(uncovered),
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
