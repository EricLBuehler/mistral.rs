"""CPU-only completion-gate regressions using synthetic metadata."""

import unittest
from unittest.mock import patch

import finalize_candidate as workflow


def metadata():
    names = [
        "serving",
        "text",
        "c1",
        "c6_c8",
        "mixed_context",
        *workflow.HOMOGENEOUS_PHASES,
    ]
    return {
        "complete": True,
        "monitor_errors": [],
        "server_shutdown": {"forced_kill": False, "returncode": 0},
        "binary_sha256": workflow.EXPECTED_BINARY,
        "binary_sha256_after": workflow.EXPECTED_BINARY,
        "phases": [{"name": name, "complete": True, "returncode": 0} for name in names],
    }


class CompletionGateTests(unittest.TestCase):
    def reject_before_manifest(self, value, message):
        with (
            patch.object(workflow, "load", return_value=value) as read,
            patch.object(workflow.compare_stage.validator, "verify_manifest") as verify,
        ):
            with self.assertRaisesRegex(ValueError, message):
                workflow.completed_run()
            read.assert_called_once_with(workflow.RUN / "metadata.json")
            verify.assert_not_called()

    def test_incomplete_rejected_before_measurement_reads(self):
        self.reject_before_manifest(
            {"complete": False}, "Wait for complete measurements"
        )

    def test_nine_unique_successful_phases_reach_manifest_validation(self):
        value = metadata()
        with (
            patch.object(workflow, "load", side_effect=[value, {}]) as read,
            patch.object(workflow.compare_stage.validator, "verify_manifest") as verify,
        ):
            self.assertIs(workflow.completed_run(), value)
            self.assertEqual(read.call_count, 2)
            verify.assert_called_once_with(workflow.RUN, {})

    def test_extra_duplicate_cannot_hide_failed_phase(self):
        value = metadata()
        value["phases"][0]["returncode"] = 1
        value["phases"].append({"name": "serving", "complete": True, "returncode": 0})
        self.reject_before_manifest(value, "Unexpected phase count")

    def test_duplicate_replacing_missing_phase_rejected(self):
        value = metadata()
        value["phases"][-1]["name"] = "serving"
        self.reject_before_manifest(value, "Duplicate phase names")


if __name__ == "__main__":
    unittest.main()
