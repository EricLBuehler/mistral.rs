import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts import verify_release_assets as verifier


def pypi_entry(filename: str) -> dict[str, str]:
    return {"filename": filename, "packagetype": "bdist_wheel"}


def expected_pypi_entries(version: str) -> list[dict[str, str]]:
    return [
        pypi_entry(f"mistralrs-{version}-cp310-abi3-macosx_11_0_arm64.whl"),
        pypi_entry(
            f"mistralrs-{version}-cp310-abi3-"
            "manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
        ),
        pypi_entry(
            f"mistralrs-{version}-cp310-abi3-"
            "manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
        ),
        pypi_entry(f"mistralrs-{version}-cp310-abi3-win_amd64.whl"),
    ]


class Pep440VersionTests(unittest.TestCase):
    def test_stable_version(self):
        self.assertEqual(verifier.pep440_version("v0.9.3"), ("0.9.3", False))

    def test_prerelease_versions(self):
        cases = {
            "v0.8.4-rc13": "0.8.4rc13",
            "v0.9.3-rc.1": "0.9.3rc1",
            "v0.9.3-alpha.2": "0.9.3a2",
            "v0.9.3-beta.4": "0.9.3b4",
            "v0.9.3-preview.3": "0.9.3rc3",
        }
        for tag, expected in cases.items():
            with self.subTest(tag=tag):
                self.assertEqual(verifier.pep440_version(tag), (expected, True))

    def test_rejects_unsupported_tag(self):
        with self.assertRaisesRegex(ValueError, "unsupported release tag"):
            verifier.pep440_version("master")


class CudaWheelTests(unittest.TestCase):
    row = {
        "cuda_asset": "131",
        "sm": "90",
        "triple": "x86_64-unknown-linux-gnu",
    }

    def test_matches_exact_stable_base_version(self):
        assets = {"mistralrs-0.9.3+cuda131.sm90-cp310-abi3-manylinux_2_17_x86_64.whl"}
        self.assertTrue(verifier.has_cuda_wheel(assets, self.row, "0.9.3"))

    def test_matches_pep440_prerelease_base_version(self):
        assets = {
            "mistralrs-0.9.3rc1+cuda131.sm90-cp310-abi3-manylinux_2_17_x86_64.whl"
        }
        self.assertTrue(verifier.has_cuda_wheel(assets, self.row, "0.9.3rc1"))

    def test_rejects_wrong_base_version(self):
        assets = {"mistralrs-0.0.0+cuda131.sm90-cp310-abi3-manylinux_2_17_x86_64.whl"}
        self.assertFalse(verifier.has_cuda_wheel(assets, self.row, "0.9.3"))

    def test_rejects_wrong_cuda_axis_or_architecture(self):
        assets = {
            "mistralrs-0.9.3+cuda130.sm90-cp310-abi3-manylinux_2_17_x86_64.whl",
            "mistralrs-0.9.3+cuda131.sm90-cp310-abi3-manylinux_2_17_aarch64.whl",
        }
        self.assertFalse(verifier.has_cuda_wheel(assets, self.row, "0.9.3"))

    def test_rejects_wrong_python_abi_or_operating_system(self):
        assets = {
            "mistralrs-0.9.3+cuda131.sm90-py2-none-manylinux_2_17_x86_64.whl",
            "mistralrs-0.9.3+cuda131.sm90-cp310-abi3-macosx_11_0_x86_64.whl",
        }
        self.assertFalse(verifier.has_cuda_wheel(assets, self.row, "0.9.3"))


class PypiWheelTests(unittest.TestCase):
    def test_requires_exact_four_platform_wheels(self):
        verified, missing, invalid = verifier.verify_pypi_wheels(
            {"urls": expected_pypi_entries("0.9.3")}, "0.9.3"
        )
        self.assertEqual(verified, 4)
        self.assertEqual(missing, [])
        self.assertEqual(invalid, [])

    def test_empty_release_does_not_pass(self):
        verified, missing, invalid = verifier.verify_pypi_wheels({"urls": []}, "0.9.3")
        self.assertEqual(verified, 0)
        self.assertEqual(len(missing), 4)
        self.assertEqual(invalid, [])

    def test_rejects_wrong_versions(self):
        verified, missing, invalid = verifier.verify_pypi_wheels(
            {"urls": expected_pypi_entries("0.0.0")}, "0.9.3"
        )
        self.assertEqual(verified, 0)
        self.assertEqual(len(missing), 4)
        self.assertEqual(len(invalid), 1)
        self.assertIn("unexpected files", invalid[0])

    def test_rejects_extra_wheel(self):
        entries = expected_pypi_entries("0.9.3")
        entries.append(entries[-1].copy())
        verified, missing, invalid = verifier.verify_pypi_wheels(
            {"urls": entries}, "0.9.3"
        )
        self.assertEqual(verified, 3)
        self.assertEqual(missing, [])
        self.assertEqual(
            invalid, ["pypi wheel: expected one Windows amd64 wheel, found 2"]
        )

    def test_rejects_wrong_python_abi(self):
        entries = [
            {
                **entry,
                "filename": entry["filename"].replace("-cp310-abi3-", "-cp399-cp399-"),
            }
            for entry in expected_pypi_entries("0.9.3")
        ]
        verified, missing, invalid = verifier.verify_pypi_wheels(
            {"urls": entries}, "0.9.3"
        )
        self.assertEqual(verified, 0)
        self.assertEqual(len(missing), 4)
        self.assertEqual(len(invalid), 1)

    def test_ignores_non_wheel_distributions(self):
        entries = expected_pypi_entries("0.9.3")
        entries.append({"filename": "mistralrs-0.9.3.tar.gz", "packagetype": "sdist"})
        verified, missing, invalid = verifier.verify_pypi_wheels(
            {"urls": entries}, "0.9.3"
        )
        self.assertEqual((verified, missing, invalid), (4, [], []))


class PypiDistHashTests(unittest.TestCase):
    def test_matches_complete_dist(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "mistralrs-0.9.3-cp310-abi3-win_amd64.whl"
            wheel.write_bytes(b"wheel")
            data = {
                "urls": [
                    {
                        **pypi_entry(wheel.name),
                        "digests": {"sha256": hashlib.sha256(b"wheel").hexdigest()},
                    }
                ]
            }
            self.assertEqual(
                verifier.verify_pypi_dist_hashes(data, [wheel]),
                (True, [], []),
            )

    def test_reports_missing_and_mismatched_files(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.whl"
            second = Path(directory) / "second.whl"
            first.write_bytes(b"local")
            second.write_bytes(b"missing")
            data = {
                "urls": [
                    {
                        **pypi_entry(first.name),
                        "digests": {"sha256": hashlib.sha256(b"remote").hexdigest()},
                    }
                ]
            }
            self.assertEqual(
                verifier.verify_pypi_dist_hashes(data, [first, second]),
                (False, [second.name], [first.name]),
            )

    def test_complete_remote_set_allows_rebuilt_wheels(self):
        self.assertTrue(
            verifier.pypi_mismatches_are_recoverable({}, ["wheel.whl"], True, [])
        )

    def test_partial_remote_set_accepts_prior_job_upload(self):
        data = {
            "urls": [
                {
                    **pypi_entry("wheel.whl"),
                    "upload_time_iso_8601": "2026-08-20T15:43:49.199477Z",
                }
            ]
        }
        prior_jobs = [
            {
                "started_at": "2026-08-20T15:43:33Z",
                "completed_at": "2026-08-20T15:44:02Z",
            }
        ]
        self.assertTrue(
            verifier.pypi_mismatches_are_recoverable(
                data, ["wheel.whl"], False, prior_jobs
            )
        )

    def test_partial_remote_set_rejects_unrelated_upload(self):
        data = {
            "urls": [
                {
                    **pypi_entry("wheel.whl"),
                    "upload_time_iso_8601": "2026-08-19T15:43:49Z",
                }
            ]
        }
        prior_jobs = [
            {
                "started_at": "2026-08-20T15:43:33Z",
                "completed_at": "2026-08-20T15:44:02Z",
            }
        ]
        self.assertFalse(
            verifier.pypi_mismatches_are_recoverable(
                data, ["wheel.whl"], False, prior_jobs
            )
        )


class DockerManifestTests(unittest.TestCase):
    def test_extracts_platforms_and_ignores_attestations(self):
        manifest = {
            "manifests": [
                {"platform": {"os": "linux", "architecture": "amd64"}},
                {"platform": {"os": "linux", "architecture": "arm64", "variant": "v8"}},
                {"platform": {"os": "unknown", "architecture": "unknown"}},
            ]
        }
        self.assertEqual(
            verifier.platforms_from_manifest(manifest),
            {"linux/amd64", "linux/arm64"},
        )

    def test_targets_follow_workflow_matrices(self):
        with open(verifier.WORKFLOW, encoding="utf-8") as workflow_file:
            jobs = yaml.safe_load(workflow_file)["jobs"]
        targets = verifier.docker_targets(
            "ghcr.io/ericlbuehler/mistral.rs",
            "0.9.3",
            jobs["docker-cuda"]["strategy"]["matrix"]["include"],
            jobs["docker-cuda-manifest"]["strategy"]["matrix"]["include"],
        )
        by_tag = dict(targets)
        self.assertEqual(len(targets), 72)
        self.assertEqual(
            by_tag["ghcr.io/ericlbuehler/mistral.rs:cpu-0.9.3"],
            {"linux/amd64", "linux/arm64"},
        )
        self.assertEqual(
            by_tag["ghcr.io/ericlbuehler/mistral.rs:cuda131-sm90-0.9.3"],
            {"linux/amd64", "linux/arm64"},
        )
        self.assertEqual(
            by_tag["ghcr.io/ericlbuehler/mistral.rs:cuda-sm90-0.9.3"],
            {"linux/amd64", "linux/arm64"},
        )
        self.assertEqual(
            by_tag["ghcr.io/ericlbuehler/mistral.rs:cuda131-sm80-0.9.3"],
            {"linux/amd64"},
        )

    def test_docker_prerelease_tag_keeps_semver_spelling(self):
        targets = verifier.docker_targets(
            "ghcr.io/ericlbuehler/mistral.rs", "0.9.3-rc.1", [], []
        )
        self.assertEqual(
            targets,
            [
                (
                    "ghcr.io/ericlbuehler/mistral.rs:cpu-0.9.3-rc.1",
                    {"linux/amd64", "linux/arm64"},
                )
            ],
        )

    def test_platform_set_mismatch_is_detectable(self):
        expected = {"linux/amd64", "linux/arm64"}
        actual = verifier.platforms_from_manifest(
            {"manifests": [{"platform": {"os": "linux", "architecture": "amd64"}}]}
        )
        self.assertNotEqual(actual, expected)

    def test_parses_raw_manifest_json(self):
        raw = json.dumps(
            {"manifests": [{"platform": {"os": "linux", "architecture": "amd64"}}]}
        )
        self.assertEqual(
            verifier.platforms_from_manifest(json.loads(raw)),
            {"linux/amd64"},
        )


if __name__ == "__main__":
    unittest.main()
