"""
OpenAI-compatible Skills with upload.

Start the server:
    mistralrs serve --agent -p 1234 -m Qwen/Qwen3-4B

Skills require the shell executor, so --enable-shell is the minimum flag.
Use --agent when you want the full agent runtime.

Then run this script:
    python examples/server/skills.py
"""

from pathlib import Path
from pprint import pprint
import tempfile
import zipfile

from openai import OpenAI


BASE_URL = "http://localhost:1234/v1"
API_KEY = "foobar"
MODEL = "default"
client = OpenAI(api_key=API_KEY, base_url=f"{BASE_URL}/")


def write_sample_skill(root: Path) -> Path:
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


def zip_skill(skill_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in skill_dir.rglob("*"):
            archive.write(path, path.relative_to(skill_dir.parent))


def upload_skill(zip_path: Path) -> dict:
    with zip_path.open("rb") as handle:
        return client.post(
            "/skills",
            cast_to=dict,
            files={"file": (zip_path.name, handle, "application/zip")},
        )


with tempfile.TemporaryDirectory() as temp_dir:
    temp_path = Path(temp_dir)
    skill_dir = write_sample_skill(temp_path)
    zip_path = temp_path / "invoice-auditor.zip"
    zip_skill(skill_dir, zip_path)
    skill = upload_skill(zip_path)

print(f"Uploaded skill: {skill['id']} ({skill['name']})")

response = client.responses.create(
    model=MODEL,
    input=(
        "Use the uploaded invoice-auditor skill. Read its instructions, run its "
        "bundled invoice check, and report the result."
    ),
    tools=[
        {
            "type": "shell",
            "environment": {
                "type": "container_auto",
                "skills": [
                    {
                        "type": "skill_reference",
                        "skill_id": skill["id"],
                        "version": "latest",
                    }
                ],
            },
        }
    ],
    tool_choice="required",
)

print("\nSkill response:")
print(response.output_text)

print("\nRaw response output:")
pprint(response.output)
