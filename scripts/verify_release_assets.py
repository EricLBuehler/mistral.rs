#!/usr/bin/env python3
"""Check a tagged release against the artifact contract implied by release.yml.

Expected tarballs, wheels, and Docker tags are derived from the workflow's own
matrices, so editing the matrix automatically updates the contract. Exits nonzero
with a list of everything missing. Usage: verify_release_assets.py <tag>
"""

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

import yaml

WORKFLOW = ".github/workflows/release.yml"
FIXED_ASSETS = [
    "mistralrs-cpu-x86_64-unknown-linux-gnu.tar.gz",
    "mistralrs-cpu-aarch64-unknown-linux-gnu.tar.gz",
    "mistralrs-cpu-aarch64-unknown-linux-gnu-v8_0.tar.gz",
    "mistralrs-cpu-x86_64-pc-windows-msvc.zip",
    "mistralrs-metal-aarch64-apple-darwin.tar.gz",
]


def main() -> int:
    tag = sys.argv[1]
    ver = tag.removeprefix("v")
    repo = os.environ.get("GITHUB_REPOSITORY", "EricLBuehler/mistral.rs")
    image = f"ghcr.io/{repo.lower()}"

    jobs = yaml.safe_load(open(WORKFLOW))["jobs"]
    cuda_rows = jobs["linux-cuda"]["strategy"]["matrix"]["include"]
    docker_rows = jobs["docker-cuda"]["strategy"]["matrix"]["include"]
    manifest_rows = jobs["docker-cuda-manifest"]["strategy"]["matrix"]["include"]

    out = subprocess.run(
        ["gh", "release", "view", tag, "-R", repo, "--json", "assets"],
        capture_output=True,
        text=True,
        check=True,
    )
    assets = {a["name"] for a in json.loads(out.stdout)["assets"]}

    missing = []

    for name in FIXED_ASSETS:
        if name not in assets:
            missing.append(f"asset: {name}")

    for row in cuda_rows:
        tarball = f"mistralrs-cuda{row['cuda_asset']}-sm{row['sm']}-{row['triple']}.tar.gz"
        if tarball not in assets:
            missing.append(f"asset: {tarball}")
        # rc tags PEP440-normalize the base version, so match on the local-version part only
        plat = row["triple"].split("-")[0]
        pat = re.compile(
            rf"mistralrs-.*\+cuda{row['cuda_asset']}\.sm{row['sm']}-.*_{plat}\.whl"
        )
        if not any(pat.fullmatch(a) for a in assets):
            missing.append(f"wheel: +cuda{row['cuda_asset']}.sm{row['sm']} ({plat})")

    docker_tags = [f"{image}:cpu-{ver}"]
    for row in docker_rows:
        docker_tags.append(f"{image}:cuda{row['cuda_asset']}-sm{row['sm']}-{ver}-{row['arch']}")
    for row in manifest_rows:
        docker_tags.append(f"{image}:cuda{row['cuda_asset']}-sm{row['sm']}-{ver}")
    for dtag in docker_tags:
        res = subprocess.run(
            ["docker", "buildx", "imagetools", "inspect", dtag],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            missing.append(f"image: {dtag}")

    if "-" not in tag:
        url = f"https://pypi.org/pypi/mistralrs/{ver}/json"
        try:
            urllib.request.urlopen(url, timeout=30)
        except urllib.error.HTTPError:
            missing.append(f"pypi: mistralrs=={ver}")

    expected = len(FIXED_ASSETS) + 2 * len(cuda_rows) + len(docker_tags) + (1 if "-" not in tag else 0)
    summary = f"{tag}: {expected - len(missing)}/{expected} artifacts published"
    print(summary)
    for item in missing:
        print(f"MISSING {item}")
    if step_summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(step_summary, "a") as f:
            f.write(f"### {summary}\n\n")
            f.writelines(f"- MISSING {item}\n" for item in missing)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
