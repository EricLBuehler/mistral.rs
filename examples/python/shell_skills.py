"""
Local shell skill mount with the Python SDK.

The request mounts a local skill directory under `skills/invoice-auditor/`.

Run with:
    pip install -e mistralrs-pyo3 --features code-execution
    python examples/python/shell_skills.py
"""

from pathlib import Path
import tempfile

from mistralrs import ChatCompletionRequest, Runner, ShellConfig, ShellSkillMount, Which


def write_invoice_skill(root: Path) -> Path:
    skill_dir = root / "invoice-auditor"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        """---
name: invoice-auditor
description: Checks invoice line items and totals with a local Python helper.
---

# Invoice Auditor

Use `python3 skills/invoice-auditor/check_invoice.py skills/invoice-auditor/invoice.csv`
to validate the bundled invoice. Report whether the declared total matches the
sum of the line items.
""",
        encoding="utf-8",
    )
    (skill_dir / "invoice.csv").write_text(
        """item,amount
hosting,25.00
storage,12.50
support,17.50
declared_total,55.00
""",
        encoding="utf-8",
    )
    (skill_dir / "check_invoice.py").write_text(
        """import csv
import sys

with open(sys.argv[1], newline="") as handle:
    rows = list(csv.DictReader(handle))

declared = float(rows[-1]["amount"])
line_total = sum(float(row["amount"]) for row in rows[:-1])
print(f"line_total={line_total:.2f}")
print(f"declared_total={declared:.2f}")
print("status=match" if line_total == declared else "status=mismatch")
""",
        encoding="utf-8",
    )
    return skill_dir


def main():
    runner = Runner(
        which=Which.Plain(model_id="Qwen/Qwen3-4B"),
        shell_config=ShellConfig(),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        skill_dir = write_invoice_skill(Path(temp_dir))
        response = runner.send_chat_completion_request(
            ChatCompletionRequest(
                model="default",
                messages=[
                    {
                        "role": "user",
                        "content": "Use the invoice-auditor skill to check the bundled invoice.",
                    }
                ],
                shell_skills=[
                    ShellSkillMount(
                        name="invoice-auditor",
                        description="Checks invoice line items and totals with a local Python helper.",
                        source_path=skill_dir,
                    )
                ],
                max_tool_rounds=6,
            )
        )

    for choice in response.choices:
        print(choice.message.content)


if __name__ == "__main__":
    main()
