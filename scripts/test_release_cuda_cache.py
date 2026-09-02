import re
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/release.yml"
BUILD_ROOT_ENV = "MISTRALRS_CUDA_BUILD_ROOT"
BUILD_ROOT = "/cache/mistralrs-cuda-objects-v1"
CACHE_IDENTITY = (
    "mistral.rs-cuda-v1-${{ matrix.runner }}-${{ matrix.cuda }}-sm${{ matrix.sm }}"
)
BUILD_SCRIPTS = {
    "mistralrs-core/build.rs": {
        "component": "core",
        "archives": ["libmistralrscuda.a", "libmistralrsflashinfergdn.a"],
    },
    "mistralrs-quant/build.rs": {
        "component": "quant",
        "archives": ["libmistralrsquant.a", "libmistralrsdeepgemm.a"],
    },
    "mistralrs-paged-attn/build.rs": {
        "component": "paged-attn",
        "archives": ["libmistralrspagedattention.a", "libmistralrsfa3paged.a"],
    },
    "mistralrs-flash-attn/build.rs": {
        "component": "flash-attn",
        "archives": ["libflashattention.a"],
    },
}


def workflow_job() -> dict:
    with WORKFLOW.open(encoding="utf-8") as workflow_file:
        return yaml.safe_load(workflow_file)["jobs"]["linux-cuda"]


def named_step(job: dict, name: str) -> dict:
    return next(step for step in job["steps"] if step.get("name") == name)


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


class ReleaseCudaCacheWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.job = workflow_job()

    def test_shared_build_root_is_mounted_cache_storage(self):
        self.assertEqual(self.job["env"][BUILD_ROOT_ENV], BUILD_ROOT)
        self.assertEqual(
            self.job["env"]["CUDAFORGE_HOME"],
            "/cache/cudaforge-dependencies-v1",
        )
        self.assertIn("/cache:/cache", self.job["container"]["volumes"])

    def test_cache_identity_has_every_binary_compatibility_axis(self):
        required = (
            "v1",
            "${{ matrix.runner }}",
            "${{ matrix.cuda }}",
            "${{ matrix.sm }}",
        )
        candidates = [
            value
            for value in strings(self.job)
            if all(axis in value for axis in required)
        ]
        self.assertTrue(candidates, "linux-cuda has no per-axis cache profile identity")
        self.assertTrue(
            all("matrix.cuda_asset" not in candidate for candidate in candidates),
            "cache identity must use the exact toolkit version, not the asset alias",
        )
        self.assertEqual(
            self.job["runs-on"],
            "${{ matrix.runner }};overrides.cache-tag=" + CACHE_IDENTITY,
        )
        self.assertEqual(self.job["concurrency"]["group"], CACHE_IDENTITY)
        self.assertFalse(self.job["concurrency"]["cancel-in-progress"])
        self.assertEqual(self.job["concurrency"]["queue"], "max")

        rows = self.job["strategy"]["matrix"]["include"]
        for candidate in candidates:
            identities = {
                candidate.replace("${{ matrix.runner }}", row["runner"])
                .replace("${{ matrix.cuda }}", row["cuda"])
                .replace("${{ matrix.sm }}", row["sm"])
                for row in rows
            }
            self.assertEqual(len(identities), len(rows))

    def test_cli_and_wheel_use_the_matrix_target(self):
        cli = named_step(self.job, "Build")["run"]
        wheel = named_step(self.job, "Build CUDA wheel")["run"]
        bundle = named_step(self.job, "Bundle runtime libraries")["run"]
        target = '--target "${{ matrix.triple }}"'
        self.assertIn(target, cli)
        self.assertIn(target, wheel)
        self.assertIn(
            "bin=target/${{ matrix.triple }}/release/mistralrs",
            bundle,
        )


class SharedCudaBuildRootTests(unittest.TestCase):
    def test_all_cuda_build_scripts_use_disjoint_shared_subdirectories(self):
        for relative_path, contract in BUILD_SCRIPTS.items():
            with self.subTest(build_script=relative_path):
                source = (ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn(BUILD_ROOT_ENV, source)
                self.assertRegex(
                    source,
                    rf"cargo:rerun-if-env-changed=(?:{BUILD_ROOT_ENV}|\{{CUDA_BUILD_ROOT_ENV\}})",
                )
                self.assertRegex(
                    source,
                    rf'join\(\s*"[^"]*{re.escape(contract["component"])}[^"]*"\s*\)',
                )
                self.assertRegex(
                    source,
                    r"std::env::var(?:_os)?\(CUDA_BUILD_ROOT_ENV\)",
                )
                self.assertGreaterEqual(source.count("cuda_build_dir("), 2)
                self.assertIn(".watch(", source)

    def test_final_archives_remain_in_cargo_out_dir(self):
        out_dir_binding = re.compile(
            r"let\s+(\w+)\s*=\s*PathBuf::from\(std::env::var\(\"OUT_DIR\"\)"
        )
        for relative_path, contract in BUILD_SCRIPTS.items():
            source = (ROOT / relative_path).read_text(encoding="utf-8")
            out_dir_names = set(out_dir_binding.findall(source))
            with self.subTest(build_script=relative_path):
                self.assertTrue(out_dir_names)
                out_dirs = "|".join(re.escape(name) for name in out_dir_names)
                self.assertNotRegex(
                    source,
                    rf"\.out_dir\(\s*&?(?:{out_dirs})\s*\)",
                )
                for archive in contract["archives"]:
                    self.assertRegex(
                        source,
                        rf"(?:{out_dirs})\.join\(\s*\"{re.escape(archive)}\"\s*\)",
                    )


if __name__ == "__main__":
    unittest.main()
