"""CPU-only runner checks with mocked processes; never starts a server or GPU tool."""

import argparse
import ast
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
from unittest.mock import Mock, patch

import observe
import run_final as runner


def main():
    checks = []
    for path in Path(__file__).parent.glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    checks.append("Python syntax")
    args = argparse.Namespace(
        port=1234, label="test", include_serving=True, include_smokes=True
    )
    jobs = runner.phases(args, Path("/scripts"), Path("/output"))
    assert [name for name, _ in jobs] == [
        "serving",
        "text",
        "c1",
        "c6_c8",
        "mixed_context",
    ]
    for name, command in jobs:
        if name not in ("c1", "c6_c8"):
            continue
        for key, expected in (
            ("--warmup", "2"),
            ("--trials", "5"),
            ("--max-tokens", "128"),
            ("--modes", "closed-loop"),
        ):
            assert command[command.index(key) + 1] == expected
        assert command[command.index("--requests") + 1] == (
            "8" if name == "c1" else "24"
        )
        index = command.index("--concurrencies") + 1
        assert command[index : command.index("--requests")] == (
            ["1"] if name == "c1" else ["6", "8"]
        )
    server_command = runner.server_command(Path("/candidate"), 1234)
    assert server_command[server_command.index("--max-seqs") + 1] == "8"
    assert server_command[server_command.index("--prefix-cache-n") + 1] == "0"
    checks.append("Exact workload counts/settings and max-seqs 8")

    with patch.dict(os.environ, {"MISTRALRS_MOE_BACKEND": "cutile"}, clear=True):
        try:
            runner.environment()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Backend override accepted")
    with patch.dict(os.environ, {"HF_TOKEN": "not-to-be-recorded"}, clear=True):
        _, recorded = runner.environment()
        assert "HF_TOKEN" not in recorded
    checks.append("Override rejection and credential exclusion")

    for forced in (False, True):
        process = Mock(pid=12345)
        process.wait.side_effect = (
            [subprocess.TimeoutExpired("mock", 1), -signal.SIGKILL]
            if forced
            else [-signal.SIGTERM]
        )
        with patch.object(runner.os, "killpg") as kill:
            result = runner.stop_process(process)
        assert result["forced_kill"] is forced
        assert [call.args[1] for call in kill.call_args_list] == (
            [signal.SIGTERM, signal.SIGKILL] if forced else [signal.SIGTERM]
        )
    checks.append("Owned group termination and forced-kill accounting")

    for kind in (
        "success",
        "request_failure",
        "server_exit",
        "monitor_exit",
        "timeout",
    ):
        with tempfile.TemporaryDirectory(
            prefix="serving-runner-self-check-"
        ) as temporary:
            output = Path(temporary)
            metadata = {"phases": []}
            child = Mock(pid=98765)
            child.returncode = 1 if kind == "request_failure" else 0
            child.poll.return_value = (
                child.returncode if kind in ("success", "request_failure") else None
            )
            server = Mock(pid=12345)
            server.poll.return_value = 1 if kind == "server_exit" else None
            gpu = Mock()
            gpu.poll.return_value = 1 if kind == "monitor_exit" else None
            with (
                patch.object(runner.subprocess, "Popen", return_value=child),
                patch.object(runner, "snapshot_phase") as snapshot,
                patch.object(
                    runner, "stop_process", return_value={"mock_cleanup": True}
                ) as cleanup,
                patch.object(
                    runner, "PHASE_TIMEOUT_SECONDS", -1 if kind == "timeout" else 7200
                ),
            ):
                try:
                    runner.run_phase(
                        kind, ["mock"], output, "unused", server, gpu, [], metadata
                    )
                except (RuntimeError, TimeoutError):
                    assert kind != "success"
                else:
                    assert kind == "success"
            record = json.loads((output / "metadata.json").read_text())["phases"][0]
            assert record["complete"] is (kind == "success")
            assert "finished" in record
            assert snapshot.call_args_list[0].args[-1] == "before"
            assert cleanup.called is (
                kind in ("server_exit", "monitor_exit", "timeout")
            )
            if kind == "success":
                assert snapshot.call_args_list[-1].args[-1] == "after"
    checks.append(
        "Phase success, request failure, server exit, monitor exit, and timeout"
    )

    own = observe.process_record(os.getpid())
    assert own["pid"] == os.getpid() and own["starttime_ticks"] > 0
    memory = observe.memory_snapshot(os.getpid())
    assert memory["page_size_bytes"] > 0 and memory["pswpin_pages"] >= 0
    assert "VmSwap_KiB" in memory["model"]
    checks.append("Linux process identity, VmSwap and global swap parsing")
    report = {
        "complete": True,
        "checks": checks,
        "server_requests": 0,
        "gpu_commands": 0,
        "builds": 0,
    }
    (Path(__file__).parent / "self_check.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
