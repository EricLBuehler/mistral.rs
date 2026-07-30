//! Local shell skill mount example.
//!
//! The request mounts a local skill directory under `skills/invoice-auditor/`.
//!
//! Run with: `cargo run --release --features code-execution --example shell_skills -p mistralrs`

use std::{env, fs, path::PathBuf};

use anyhow::Result;
use mistralrs::{
    IsqBits, ModelBuilder, RequestBuilder, ShellConfig, TextMessageRole, TextMessages,
};

fn write_invoice_skill(root: PathBuf) -> Result<PathBuf> {
    let skill_dir = root.join("invoice-auditor");
    fs::create_dir_all(&skill_dir)?;
    fs::write(
        skill_dir.join("SKILL.md"),
        r#"---
name: invoice-auditor
description: Checks invoice line items and totals with a local Python helper.
---

# Invoice Auditor

Use `python3 skills/invoice-auditor/check_invoice.py skills/invoice-auditor/invoice.csv`
to validate the bundled invoice. Report whether the declared total matches the
sum of the line items.
"#,
    )?;
    fs::write(
        skill_dir.join("invoice.csv"),
        "item,amount\nhosting,25.00\nstorage,12.50\nsupport,17.50\ndeclared_total,55.00\n",
    )?;
    fs::write(
        skill_dir.join("check_invoice.py"),
        r#"import csv
import sys

with open(sys.argv[1], newline="") as handle:
    rows = list(csv.DictReader(handle))

declared = float(rows[-1]["amount"])
line_total = sum(float(row["amount"]) for row in rows[:-1])
print(f"line_total={line_total:.2f}")
print(f"declared_total={declared:.2f}")
print("status=match" if line_total == declared else "status=mismatch")
"#,
    )?;
    Ok(skill_dir)
}

#[tokio::main]
async fn main() -> Result<()> {
    let skill_dir = write_invoice_skill(
        env::temp_dir().join(format!("mistralrs-shell-skill-{}", std::process::id())),
    )?;

    let model = ModelBuilder::new("google/gemma-4-E4B-it")
        .with_auto_isq(IsqBits::Four)
        .with_logging()
        .with_shell_execution(ShellConfig::default())
        .build()
        .await?;

    let messages = TextMessages::new().add_message(
        TextMessageRole::User,
        "Use the invoice-auditor skill to check the bundled invoice.",
    );
    let request = RequestBuilder::from(messages)
        .with_shell_skill(
            "invoice-auditor",
            "Checks invoice line items and totals with a local Python helper.",
            skill_dir,
        )
        .with_max_tool_rounds(6);

    let response = model.send_chat_request(request).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());

    Ok(())
}
