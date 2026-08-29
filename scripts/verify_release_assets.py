#!/usr/bin/env python3
"""Check a tagged release against the artifact contract implied by release.yml.

Expected tarballs, wheels, and Docker tags are derived from the workflow's own
matrices, so editing the matrix automatically updates the contract. Exits nonzero
with a list of everything missing or invalid. Usage: verify_release_assets.py <tag>
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import yaml

WORKFLOW = ".github/workflows/release.yml"
FIXED_ASSETS = [
    "mistralrs-cpu-x86_64-unknown-linux-gnu.tar.gz",
    "mistralrs-cpu-aarch64-unknown-linux-gnu.tar.gz",
    "mistralrs-cpu-aarch64-unknown-linux-gnu-v8_0.tar.gz",
    "mistralrs-cpu-x86_64-pc-windows-msvc.zip",
    "mistralrs-metal-aarch64-apple-darwin.tar.gz",
]
PYPI_PLATFORMS = (
    "macOS arm64",
    "manylinux x86_64",
    "manylinux aarch64",
    "Windows amd64",
)
PYTHON_TAG = "cp310"
ABI_TAG = "abi3"
PYPI_UPLOAD_CLOCK_SKEW_SECONDS = 60
RELEASE_VERSION_RE = re.compile(
    r"^v?(?P<release>\d+\.\d+\.\d+)"
    r"(?:(?:[-_.]?)(?P<phase>alpha|a|beta|b|rc|c|pre|preview)(?:[-_.]?)(?P<number>\d+)?)?$",
    re.IGNORECASE,
)
WHEEL_RE = re.compile(
    r"^mistralrs-(?P<version>[^-]+)-(?P<python>[^-]+)-(?P<abi>[^-]+)-(?P<platform>[^-]+)\.whl$"
)


def pep440_version(tag: str) -> tuple[str, bool]:
    match = RELEASE_VERSION_RE.fullmatch(tag)
    if match is None:
        raise ValueError(f"unsupported release tag: {tag}")
    phase = match.group("phase")
    if phase is None:
        return match.group("release"), False
    canonical_phase = {
        "a": "a",
        "alpha": "a",
        "b": "b",
        "beta": "b",
        "c": "rc",
        "pre": "rc",
        "preview": "rc",
        "rc": "rc",
    }[phase.lower()]
    number = int(match.group("number") or 0)
    return f"{match.group('release')}{canonical_phase}{number}", True


def wheel_parts(filename: str) -> dict[str, str] | None:
    match = WHEEL_RE.fullmatch(filename)
    return match.groupdict() if match else None


def has_cuda_wheel(assets: set[str], row: dict[str, str], version: str) -> bool:
    expected_version = f"{version}+cuda{row['cuda_asset']}.sm{row['sm']}"
    arch = row["triple"].split("-")[0]
    expected_platform = {
        "aarch64": "manylinux aarch64",
        "x86_64": "manylinux x86_64",
    }[arch]
    for asset in assets:
        parts = wheel_parts(asset)
        if parts is None:
            continue
        if (
            parts["version"] == expected_version
            and parts["python"] == PYTHON_TAG
            and parts["abi"] == ABI_TAG
            and pypi_platform(parts["platform"]) == expected_platform
        ):
            return True
    return False


def pypi_platform(platform_tag: str) -> str | None:
    tags = platform_tag.split(".")
    if any(tag.startswith("macosx_") and tag.endswith("_arm64") for tag in tags):
        return "macOS arm64"
    if any(tag.startswith("manylinux") and tag.endswith("_x86_64") for tag in tags):
        return "manylinux x86_64"
    if any(tag.startswith("manylinux") and tag.endswith("_aarch64") for tag in tags):
        return "manylinux aarch64"
    if "win_amd64" in tags:
        return "Windows amd64"
    return None


def verify_pypi_wheels(data: dict, version: str) -> tuple[int, list[str], list[str]]:
    wheels = [
        entry["filename"]
        for entry in data.get("urls", [])
        if entry.get("packagetype") == "bdist_wheel" and "filename" in entry
    ]
    matches = {platform: [] for platform in PYPI_PLATFORMS}
    unexpected = []
    for filename in wheels:
        parts = wheel_parts(filename)
        platform = pypi_platform(parts["platform"]) if parts else None
        if (
            parts is None
            or parts["version"] != version
            or parts["python"] != PYTHON_TAG
            or parts["abi"] != ABI_TAG
            or platform is None
        ):
            unexpected.append(filename)
            continue
        matches[platform].append(filename)

    missing = []
    invalid = []
    verified = 0
    for platform, filenames in matches.items():
        if len(filenames) == 1:
            verified += 1
        elif not filenames:
            missing.append(f"pypi wheel: mistralrs=={version} ({platform})")
        else:
            invalid.append(
                f"pypi wheel: expected one {platform} wheel, found {len(filenames)}"
            )
    if unexpected:
        invalid.append(f"pypi wheel: unexpected files: {', '.join(sorted(unexpected))}")
    return verified, missing, invalid


def verify_pypi_dist_hashes(
    data: dict, wheels: list[Path]
) -> tuple[bool, list[str], list[str]]:
    local = {wheel.name: wheel for wheel in wheels}
    remote = {
        entry["filename"]: entry.get("digests", {}).get("sha256")
        for entry in data.get("urls", [])
        if entry.get("packagetype") == "bdist_wheel" and "filename" in entry
    }
    missing = sorted(local.keys() - remote.keys())
    mismatched = []
    for name in sorted(local.keys() & remote.keys()):
        with local[name].open("rb") as wheel_file:
            digest = hashlib.file_digest(wheel_file, "sha256").hexdigest()
        if not remote[name] or digest != remote[name]:
            mismatched.append(name)
    return not missing, missing, mismatched


def pypi_mismatches_are_recoverable(
    data: dict,
    mismatched: list[str],
    complete: bool,
    prior_jobs: list[dict[str, str]],
) -> bool:
    if not mismatched or complete:
        return True

    def parse_timestamp(value: str | None) -> datetime | None:
        if value is None:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    windows = []
    clock_skew = timedelta(seconds=PYPI_UPLOAD_CLOCK_SKEW_SECONDS)
    for job in prior_jobs:
        started_at = parse_timestamp(job.get("started_at"))
        completed_at = parse_timestamp(job.get("completed_at"))
        if started_at is not None and completed_at is not None:
            windows.append((started_at - clock_skew, completed_at + clock_skew))

    uploaded_at = {
        entry["filename"]: parse_timestamp(entry.get("upload_time_iso_8601"))
        for entry in data.get("urls", [])
        if entry.get("packagetype") == "bdist_wheel" and "filename" in entry
    }
    return all(
        uploaded_at.get(filename) is not None
        and any(start <= uploaded_at[filename] <= end for start, end in windows)
        for filename in mismatched
    )


def docker_targets(
    image: str,
    version: str,
    docker_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
) -> list[tuple[str, set[str]]]:
    targets = [(f"{image}:cpu-{version}", {"linux/amd64", "linux/arm64"})]
    for row in docker_rows:
        tag = f"{image}:cuda{row['cuda_asset']}-sm{row['sm']}-{version}-{row['arch']}"
        targets.append((tag, {row["platform"]}))
    for row in manifest_rows:
        tag = f"{image}:cuda{row['cuda_asset']}-sm{row['sm']}-{version}"
        platforms = {f"linux/{arch}" for arch in row["arches"].split()}
        targets.append((tag, platforms))
        if row["cuda_asset"] == "131":
            targets.append((f"{image}:cuda-sm{row['sm']}-{version}", platforms))
    return targets


def platforms_from_manifest(manifest: dict) -> set[str]:
    platforms = set()
    for descriptor in manifest.get("manifests", []):
        platform = descriptor.get("platform", {})
        os_name = platform.get("os")
        arch = platform.get("architecture")
        if os_name and arch and os_name != "unknown" and arch != "unknown":
            platforms.add(f"{os_name}/{arch}")
    return platforms


def inspect_docker_platforms(tag: str) -> tuple[set[str] | None, str | None]:
    result = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", "--raw", tag],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None, result.stderr.strip()
    try:
        manifest = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        return None, f"invalid manifest JSON: {error}"
    return platforms_from_manifest(manifest), None


def main() -> int:
    tag = sys.argv[1]
    version, prerelease = pep440_version(tag)
    docker_version = tag.removeprefix("v")
    repo = os.environ.get("GITHUB_REPOSITORY", "EricLBuehler/mistral.rs")
    image = f"ghcr.io/{repo.lower()}"

    with open(WORKFLOW, encoding="utf-8") as workflow_file:
        jobs = yaml.safe_load(workflow_file)["jobs"]
    cuda_rows = jobs["linux-cuda"]["strategy"]["matrix"]["include"]
    docker_rows = jobs["docker-cuda"]["strategy"]["matrix"]["include"]
    manifest_rows = jobs["docker-cuda-manifest"]["strategy"]["matrix"]["include"]

    output = subprocess.run(
        ["gh", "release", "view", tag, "-R", repo, "--json", "assets"],
        capture_output=True,
        text=True,
        check=True,
    )
    assets = {asset["name"] for asset in json.loads(output.stdout)["assets"]}

    missing = []
    invalid = []
    verified = 0

    for name in FIXED_ASSETS:
        if name in assets:
            verified += 1
        else:
            missing.append(f"asset: {name}")

    for row in cuda_rows:
        tarball = (
            f"mistralrs-cuda{row['cuda_asset']}-sm{row['sm']}-{row['triple']}.tar.gz"
        )
        if tarball in assets:
            verified += 1
        else:
            missing.append(f"asset: {tarball}")
        if has_cuda_wheel(assets, row, version):
            verified += 1
        else:
            arch = row["triple"].split("-")[0]
            local = f"cuda{row['cuda_asset']}.sm{row['sm']}"
            missing.append(f"wheel: mistralrs-{version}+{local} (*_{arch}.whl)")

    targets = docker_targets(image, docker_version, docker_rows, manifest_rows)
    for docker_tag, expected_platforms in targets:
        actual_platforms, error = inspect_docker_platforms(docker_tag)
        if error is not None:
            missing.append(f"image: {docker_tag} ({error})")
        elif actual_platforms != expected_platforms:
            expected = ", ".join(sorted(expected_platforms))
            actual = ", ".join(sorted(actual_platforms or set())) or "none"
            invalid.append(
                f"image: {docker_tag} platforms expected [{expected}], found [{actual}]"
            )
        else:
            verified += 1

    expected = len(FIXED_ASSETS) + 2 * len(cuda_rows) + len(targets)
    if not prerelease:
        expected += len(PYPI_PLATFORMS)
        url = f"https://pypi.org/pypi/mistralrs/{version}/json"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                pypi_data = json.load(response)
        except urllib.error.HTTPError:
            missing.append(f"pypi: mistralrs=={version}")
        else:
            pypi_verified, pypi_missing, pypi_invalid = verify_pypi_wheels(
                pypi_data, version
            )
            verified += pypi_verified
            missing.extend(pypi_missing)
            invalid.extend(pypi_invalid)

    problems = len(missing) + len(invalid)
    summary = f"{tag}: {verified}/{expected} required artifacts verified"
    if problems:
        summary += f"; {problems} problem(s)"
    print(summary)
    for item in missing:
        print(f"MISSING {item}")
    for item in invalid:
        print(f"INVALID {item}")
    if step_summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(step_summary, "a", encoding="utf-8") as summary_file:
            summary_file.write(f"### {summary}\n\n")
            summary_file.writelines(f"- MISSING {item}\n" for item in missing)
            summary_file.writelines(f"- INVALID {item}\n" for item in invalid)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
