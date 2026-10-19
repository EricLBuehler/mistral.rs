import io
import json
import os
from pathlib import Path
import pty
import re
import select
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import termios
import textwrap
import time
import unittest


INSTALLER = Path(__file__).resolve().parents[2] / "install.sh"
CPU_ASSET = "mistralrs-cpu-x86_64-unknown-linux-gnu.tar.gz"
RELEASES_URL = "https://github.com/EricLBuehler/mistral.rs/releases"
COMMAND_TIMEOUT_SECONDS = 10
LONG_RUNNING_COMMAND_SECONDS = 30
PROGRESS_SAMPLE_SECONDS = 1
DIAGNOSTIC_CONTEXT_LINES = 40
UNICODE_DIAGNOSTIC = "error: caf\u00e9 \u65e5\u672c\u8a9e \u03bb"
PTY_READ_SIZE = 65536
PTY_POLL_SECONDS = 0.02
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
MENU_READY = re.escape("Up/Down to move, Enter to select, or press a shortcut.")
SYSTEM_COMMANDS = (
    "awk",
    "basename",
    "cat",
    "chmod",
    "cp",
    "cut",
    "date",
    "dd",
    "dirname",
    "env",
    "grep",
    "gzip",
    "head",
    "id",
    "ln",
    "ls",
    "mkdir",
    "mkfifo",
    "mktemp",
    "mv",
    "readlink",
    "rm",
    "rmdir",
    "sed",
    "sh",
    "sleep",
    "sort",
    "stty",
    "tail",
    "tar",
    "tee",
    "touch",
    "tr",
    "wc",
    "xargs",
)
STUB_COMMANDS = (
    "apt-get",
    "cargo",
    "curl",
    "nvidia-smi",
    "nvcc",
    "rustc",
    "sudo",
    "uname",
    "xcode-select",
    "xcodebuild",
    "xcrun",
)
STUB_SCRIPT = """\
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

command = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["INSTALL_TEST_LOG"], "a") as log:
    log.write(json.dumps([command, *args]) + "\\n")

if command == "uname":
    print(os.environ.get("INSTALL_TEST_OS", "Linux") if "-s" in args
          else os.environ.get("INSTALL_TEST_ARCH", "x86_64"))
elif command == "nvidia-smi":
    if os.environ.get("INSTALL_TEST_NVIDIA") != "1":
        sys.exit(1)
    if "--query-gpu=compute_cap" in args:
        print("8.9")
    else:
        print("NVIDIA-SMI 580.0 Driver Version: 580.0 CUDA Version: 13.1")
elif command == "curl":
    if "-w" in args:
        print("https://github.com/EricLBuehler/mistral.rs/releases/tag/v0.9.5", end="")
    elif any(arg.startswith("-") and "I" in arg for arg in args):
        size = Path(os.environ["INSTALL_TEST_ARCHIVE"]).stat().st_size
        print(f"HTTP/2 200\\r\\nContent-Length: {size}\\r\\n")
    elif "-o" in args:
        if os.environ.get("INSTALL_TEST_DOWNLOAD_FAIL") == "1":
            print("curl: simulated download failure", file=sys.stderr)
            sys.exit(22)
        destination = args[args.index("-o") + 1]
        if os.environ.get("INSTALL_TEST_DOWNLOAD_DELAY"):
            data = Path(os.environ["INSTALL_TEST_ARCHIVE"]).read_bytes()
            midpoint = len(data) // 2
            with open(destination, "wb") as output:
                output.write(data[:midpoint])
                output.flush()
                time.sleep(float(os.environ["INSTALL_TEST_DOWNLOAD_DELAY"]))
                output.write(data[midpoint:])
        else:
            shutil.copyfile(os.environ["INSTALL_TEST_ARCHIVE"], destination)
    else:
        print("Unexpected network request", file=sys.stderr)
        sys.exit(93)
elif command == "sudo":
    sys.exit(subprocess.run(args, check=False).returncode)
elif command == "apt-get":
    print("Simulated package-manager failure", file=sys.stderr)
    sys.exit(1)
elif command == "mv":
    result = subprocess.run([os.environ["INSTALL_TEST_REAL_MV"], *args], check=False)
    managed = str(Path(os.environ["HOME"]) / ".mistralrs")
    if result.returncode == 0 and args[0] == managed and args[1].startswith(managed + "-backup."):
        os.kill(os.getppid(), signal.SIGINT)
    sys.exit(result.returncode)
elif command == "rustc":
    print("rustc 1.94.0")
elif command == "nvcc" and os.environ.get("INSTALL_TEST_CUDA_TOOLKIT") == "1":
    print("Cuda compilation tools, release 13.1, V13.1.123")
elif command == "cargo" and os.environ.get("INSTALL_TEST_ALLOW_BUILD") == "1":
    Path(os.environ["INSTALL_TEST_COMMAND_PID"]).write_text(str(os.getpid()))
    if os.environ.get("INSTALL_TEST_LOCALE_LOG"):
        Path(os.environ["INSTALL_TEST_LOCALE_LOG"]).write_text(json.dumps({key: os.environ.get(key) for key in ("LC_ALL", "LANG")}))
    print("Compiling installer-fixture v0.9.5", file=sys.stderr, flush=True)
    if os.environ.get("INSTALL_TEST_BUILD_GATE"):
        while not Path(os.environ["INSTALL_TEST_BUILD_GATE"]).exists():
            time.sleep(float(os.environ["INSTALL_TEST_POLL_INTERVAL"]))
    if os.environ.get("INSTALL_TEST_VERBOSE_LINES") == "1":
        print("fixture stdout live marker", flush=True)
        print("fixture stderr live marker", file=sys.stderr, flush=True)
    if os.environ.get("INSTALL_TEST_BUILD_COUNTERS") == "1":
        for completed in (3, 8):
            print(f"\\r    Building [====>     ] {completed}/20: installer-fixture\\x1b[K", end="", file=sys.stderr, flush=True)
            time.sleep(float(os.environ["INSTALL_TEST_COUNTER_DELAY"]))
        print(file=sys.stderr)
    time.sleep(float(os.environ.get("INSTALL_TEST_BUILD_DELAY", "0")))
    if os.environ.get("INSTALL_TEST_DIAGNOSTIC_CONTEXT_LINES"):
        print("fixture stdout before failure", flush=True)
        print("\\r    Building [====>     ] 3/20: fixture-progress\\x1b[K\\r", end="", file=sys.stderr)
        print("\\x1b[31merror[E0425]: cannot find value fixture_symbol\\x1b[0m\\r", file=sys.stderr)
        print(" --> src/fixture.rs:7:9", file=sys.stderr)
        for index in range(int(os.environ["INSTALL_TEST_DIAGNOSTIC_CONTEXT_LINES"])):
            print(f"fixture diagnostic context {index:03d}", file=sys.stderr)
        if os.environ.get("INSTALL_TEST_UNICODE_DIAGNOSTIC"):
            print("\\x1b[31m" + os.environ["INSTALL_TEST_UNICODE_DIAGNOSTIC"] + "\\x1b[0m\\r", file=sys.stderr)
        print("fixture diagnostic final context", file=sys.stderr)
    if os.environ.get("INSTALL_TEST_BUILD_FAIL") == "1":
        print("error: simulated Cargo build failure", file=sys.stderr)
        sys.exit(1)
    binary = Path(args[args.index("--root") + 1]) / "bin" / "mistralrs"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\\nprintf 'mistralrs 0.9.5\\\\n'\\n")
    binary.chmod(0o755)
else:
    print("Unexpected command: " + command, file=sys.stderr)
    sys.exit(94)
"""


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mistralrs-installer-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.home = self.root / "home"
        self.bin = self.root / "bin"
        self.tmp = self.root / "tmp"
        for directory in (self.home, self.bin, self.tmp):
            directory.mkdir()
        self.log = self.root / "commands.jsonl"
        self.command_pid = self.root / "command.pid"
        self.archive = self.root / "prebuilt.tar.gz"
        self.write_archive(b"#!/bin/sh\nprintf 'mistralrs 0.9.5\\n'\n")
        for command in SYSTEM_COMMANDS:
            executable = shutil.which(command)
            self.assertIsNotNone(executable, f"Missing test utility: {command}")
            (self.bin / command).symlink_to(executable)
        for command in STUB_COMMANDS:
            script = self.bin / command
            script.write_text(f"#!{sys.executable}\n" + textwrap.dedent(STUB_SCRIPT))
            script.chmod(0o755)
        self.env = {
            "HOME": str(self.home),
            "PATH": str(self.bin),
            "TMPDIR": str(self.tmp),
            "SHELL": "/bin/sh",
            "TERM": "dumb",
            "NO_COLOR": "1",
            "LC_ALL": "C",
            "INSTALL_TEST_ARCHIVE": str(self.archive),
            "INSTALL_TEST_LOG": str(self.log),
            "INSTALL_TEST_COMMAND_PID": str(self.command_pid),
        }

    def run_installer(self, *args):
        result = subprocess.run(
            ["/bin/sh", str(INSTALLER), *args],
            cwd=self.root,
            env=self.env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            start_new_session=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
        result.raw_stdout = result.stdout.decode()
        result.stdout = ANSI_ESCAPE.sub("", result.raw_stdout)
        return result

    def commands(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def run_installer_with_edit_during_build(self, script):
        private_installer = self.root / "install-copy.sh"
        private_installer.write_text(script)
        gate = self.root / "continue-build"
        env = {
            **self.env,
            "INSTALL_TEST_ALLOW_BUILD": "1",
            "INSTALL_TEST_BUILD_GATE": str(gate),
            "INSTALL_TEST_POLL_INTERVAL": str(PTY_POLL_SECONDS),
        }
        command = [
            "/bin/sh",
            str(private_installer),
            "--yes",
            "--from-source",
            "--release",
            "master",
            "--backend",
            "cpu",
        ]
        process = subprocess.Popen(
            command,
            cwd=self.root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + COMMAND_TIMEOUT_SECONDS
        try:
            while not self.command_pid.exists():
                if process.poll() is not None:
                    self.fail(
                        "Installer exited before build: "
                        + process.stdout.read().decode()
                    )
                if time.monotonic() >= deadline:
                    self.fail("Installer did not reach the build before timeout")
                time.sleep(PTY_POLL_SECONDS)
            with private_installer.open("r+b") as installer:
                installer.write(script.encode() + b"\n`\n")
                installer.truncate()
            gate.touch()
            output, _ = process.communicate(timeout=COMMAND_TIMEOUT_SECONDS)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        result = subprocess.CompletedProcess(
            command, process.returncode, ANSI_ESCAPE.sub("", output.decode())
        )
        result.raw_stdout = output.decode()
        return result

    def write_archive(self, payload):
        with tarfile.open(self.archive, "w:gz") as archive:
            entry = tarfile.TarInfo("mistralrs")
            entry.size = len(payload)
            entry.mode = 0o755
            archive.addfile(entry, io.BytesIO(payload))

    def run_interactive(self, steps, *args, piped=False, no_color=False):
        env = {**self.env, "TERM": "xterm-256color"}
        if not no_color:
            env.pop("NO_COLOR", None)
        command = ["/bin/sh", str(INSTALLER), *args]
        if piped:
            env["INSTALL_TEST_SCRIPT"] = str(INSTALLER)
            command = [
                "/bin/sh",
                "-c",
                'cat "$INSTALL_TEST_SCRIPT" | sh -s -- "$@"',
                "sh",
                *args,
            ]
        gate_read, gate_write = os.pipe()
        child, master = pty.fork()
        if child == 0:
            os.close(gate_write)
            os.read(gate_read, 1)
            os.close(gate_read)
            os.chdir(self.root)
            os.execve(command[0], command, env)
        os.close(gate_read)
        terminal_before = termios.tcgetattr(master)
        os.write(gate_write, b"x")
        os.close(gate_write)
        chunks = []
        status = None
        next_step = 0
        cursor = 0
        deadline = time.monotonic() + COMMAND_TIMEOUT_SECONDS
        try:
            while True:
                output = ANSI_ESCAPE.sub("", b"".join(chunks).decode(errors="replace"))
                if next_step < len(steps):
                    expected, keys, *input_mode = steps[next_step]
                    canonical = bool(termios.tcgetattr(master)[3] & termios.ICANON)
                    needs_line_input = (
                        input_mode[0]
                        if input_mode
                        else keys.endswith("\n") and len(keys) > 1
                    )
                    if (
                        re.search(expected, output[cursor:])
                        and canonical == needs_line_input
                    ):
                        os.write(master, keys.encode())
                        cursor = len(output)
                        next_step += 1
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.fail(f"Installer timed out at step {next_step}: {output}")
                if not select.select(
                    [master], [], [], min(remaining, PTY_POLL_SECONDS)
                )[0]:
                    continue
                try:
                    chunk = os.read(master, PTY_READ_SIZE)
                except OSError:
                    break
                if not chunk:
                    break
                chunks.append(chunk)
            terminal_after = termios.tcgetattr(master)
            os.close(master)
            master = None
            _, status = os.waitpid(child, 0)
        finally:
            if master is not None:
                os.close(master)
            if status is None:
                os.killpg(child, signal.SIGKILL)
                os.waitpid(child, 0)
        raw_output = b"".join(chunks).decode(errors="replace")
        output = ANSI_ESCAPE.sub("", raw_output)
        self.assertEqual(next_step, len(steps), output)
        self.assertEqual(
            terminal_after,
            terminal_before,
            "Installer did not restore terminal settings",
        )
        result = subprocess.CompletedProcess(
            command, os.waitstatus_to_exitcode(status), output
        )
        result.raw_stdout = raw_output
        return result

    def download_urls(self):
        return [
            arg
            for command in self.commands()
            if command[0] == "curl" and "-o" in command
            for arg in command[1:]
            if arg.startswith("https://")
        ]

    def terminal_screen(self, output):
        rows = [""]
        row = 0
        column = 0
        offset = 0
        while offset < len(output):
            escape = ANSI_ESCAPE.match(output, offset)
            if escape:
                sequence = escape.group()
                move_up = re.fullmatch(r"\x1b\[([0-9]+)A", sequence)
                if move_up:
                    row = max(0, row - int(move_up.group(1)))
                elif sequence == "\x1b[J":
                    rows[row] = rows[row][:column]
                    del rows[row + 1 :]
                offset = escape.end()
                continue
            char = output[offset]
            offset += 1
            if char == "\r":
                column = 0
            elif char == "\n":
                row += 1
                if row == len(rows):
                    rows.append("")
            else:
                rows[row] = (
                    rows[row].ljust(column)[:column] + char + rows[row][column + 1 :]
                )
                column += 1
        return "\n".join(rows)

    def assert_no_installation(self):
        self.assertEqual(list(self.home.iterdir()), [])
        self.assertEqual(list(self.tmp.iterdir()), [])

    def assert_installed(self, result):
        self.assertEqual(result.returncode, 0, result.stdout)
        binary = self.home / ".mistralrs" / "mistralrs"
        self.assertTrue(binary.is_file(), result.stdout)
        self.assertTrue(os.access(binary, os.X_OK))
        self.assertEqual((self.home / ".local/bin/mistralrs").resolve(), binary)
        self.assertTrue((self.home / ".mistralrs/env").is_file())
        self.assertIn('. "$HOME/.mistralrs/env"', (self.home / ".profile").read_text())

    def assert_stage_order(self, result, operation):
        stages = re.findall(
            r"\[([1-4])/4\]\s+(Prepare installation|Download binary|Build from source|Verify binary|Finish setup)",
            result.stdout,
        )
        self.assertEqual(
            stages,
            [
                ("1", "Prepare installation"),
                ("2", operation),
                ("3", "Verify binary"),
                ("4", "Finish setup"),
            ],
            result.stdout,
        )

    def assert_diagnostic_log(self, result, expected_content=None):
        logs = list(self.tmp.glob("mistralrs-install.*"))
        self.assertEqual(len(logs), 1, result.stdout)
        self.assertTrue(logs[0].is_file())
        self.assertIn(str(logs[0]), result.stdout)
        if expected_content:
            self.assertIn(expected_content, logs[0].read_text())
        self.assertEqual(list(self.tmp.iterdir()), logs)
        return logs[0]

    def use_utf8_locale(self, lang_only):
        locale_command = shutil.which("locale")
        if locale_command is None:
            self.skipTest("The locale utility is required to find a UTF-8 locale")
        available = subprocess.run(
            [locale_command, "-a"],
            capture_output=True,
            text=True,
            check=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        ).stdout.splitlines()
        candidates = [
            name
            for name in available
            if "utf" in name.lower() and not name.lower().startswith("c.")
        ]
        preferred = [name for name in candidates if name.lower().startswith("en_us.")]
        if not candidates:
            self.skipTest("A non-C UTF-8 locale is required for this regression")
        locale_name = (preferred or candidates)[0]
        self.env["LANG"] = locale_name
        if lang_only:
            self.env.pop("LC_ALL", None)
        else:
            self.env["LC_ALL"] = locale_name
        self.env["INSTALL_TEST_LOCALE_LOG"] = str(self.root / "cargo-locale.json")
        return locale_name

    def assert_cargo_locale_preserved(self, locale_name, lang_only):
        inherited = json.loads((self.root / "cargo-locale.json").read_text())
        self.assertEqual(inherited["LANG"], locale_name)
        self.assertEqual(inherited["LC_ALL"], None if lang_only else locale_name)

    def check_utf8_cargo_counters(self, lang_only):
        locale_name = self.use_utf8_locale(lang_only)
        self.env.update(
            INSTALL_TEST_ALLOW_BUILD="1",
            INSTALL_TEST_BUILD_COUNTERS="1",
            INSTALL_TEST_COUNTER_DELAY=str(PROGRESS_SAMPLE_SECONDS),
        )
        result = self.run_interactive(
            [], "--yes", "--from-source", "--release", "master", "--backend", "cpu"
        )
        self.assert_installed(result)
        self.assertRegex(result.stdout, r"\b8/20\b")
        self.assertNotIn("RE error", result.stdout)
        self.assert_cargo_locale_preserved(locale_name, lang_only)
        self.assertEqual(list(self.tmp.iterdir()), [])

    def check_utf8_failure_diagnostics(self, lang_only):
        locale_name = self.use_utf8_locale(lang_only)
        self.env.update(
            INSTALL_TEST_ALLOW_BUILD="1",
            INSTALL_TEST_BUILD_FAIL="1",
            INSTALL_TEST_DIAGNOSTIC_CONTEXT_LINES=str(DIAGNOSTIC_CONTEXT_LINES),
            INSTALL_TEST_UNICODE_DIAGNOSTIC=UNICODE_DIAGNOSTIC,
        )
        result = self.run_installer(
            "--yes", "--from-source", "--release", "master", "--backend", "cpu"
        )
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(UNICODE_DIAGNOSTIC, result.stdout)
        for index in range(DIAGNOSTIC_CONTEXT_LINES):
            self.assertIn(f"fixture diagnostic context {index:03d}", result.stdout)
        self.assertNotIn("RE error", result.stdout)
        self.assertNotIn("\x1b", result.raw_stdout)
        self.assertNotIn("\r", result.raw_stdout)
        self.assert_diagnostic_log(result, UNICODE_DIAGNOSTIC)
        self.assert_cargo_locale_preserved(locale_name, lang_only)

    def test_help_is_available_without_installing(self):
        result = self.run_installer("--help")
        self.assertEqual(result.returncode, 0, result.stdout)
        for flag in (
            "--yes",
            "--dry-run",
            "--release",
            "--backend",
            "--from-source",
            "--verbose",
        ):
            self.assertIn(flag, result.stdout)
        self.assertEqual(self.commands(), [])
        self.assert_no_installation()

    def test_invalid_arguments_fail_before_detection_or_mutation(self):
        cases = (
            ("--unknown",),
            ("--release",),
            ("--backend",),
            ("--backend", "invalid"),
            ("--release", "../../invalid"),
        )
        for args in cases:
            with self.subTest(args=args):
                result = self.run_installer(*args)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertEqual(self.commands(), [])
                self.assert_no_installation()

    def test_dry_run_plans_without_downloading_or_changing_home(self):
        result = self.run_installer("--yes", "--dry-run", "--backend", "cpu")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("cpu", result.stdout.lower())
        self.assertFalse(
            any(
                command[0] in ("curl", "cargo", "apt-get")
                for command in self.commands()
            )
        )
        self.assert_no_installation()

    def test_cpu_backend_skips_cuda_on_nvidia_machine(self):
        self.env["INSTALL_TEST_NVIDIA"] = "1"
        result = self.run_installer("--yes", "--backend", "cpu")
        self.assert_installed(result)
        self.assertEqual(
            self.download_urls(), [f"{RELEASES_URL}/latest/download/{CPU_ASSET}"]
        )
        self.assertFalse(any(command[0] == "cargo" for command in self.commands()))

    def test_pinned_release_uses_requested_asset_url(self):
        result = self.run_installer("--yes", "--backend", "cpu", "--release", "v0.9.5")
        self.assert_installed(result)
        self.assertEqual(
            self.download_urls(), [f"{RELEASES_URL}/download/v0.9.5/{CPU_ASSET}"]
        )

    def test_existing_updater_environment_still_installs_noninteractively(self):
        self.env.update(
            MISTRALRS_INSTALL_YES="1",
            MISTRALRS_INSTALL_IGNORE_FFMPEG="1",
            MISTRALRS_INSTALL_TAG="v0.9.5",
        )
        result = self.run_installer()
        self.assert_installed(result)
        self.assertEqual(
            self.download_urls(), [f"{RELEASES_URL}/download/v0.9.5/{CPU_ASSET}"]
        )

    def test_apple_silicon_automatically_selects_metal(self):
        self.env.update(INSTALL_TEST_OS="Darwin", INSTALL_TEST_ARCH="arm64")
        result = self.run_installer("--yes")
        self.assert_installed(result)
        self.assertEqual(
            self.download_urls(),
            [
                f"{RELEASES_URL}/latest/download/mistralrs-metal-aarch64-apple-darwin.tar.gz"
            ],
        )

    def test_source_build_dry_run_accepts_explicit_options_without_building(self):
        result = self.run_installer(
            "--yes",
            "--dry-run",
            "--from-source",
            "--release",
            "master",
            "--backend",
            "cpu",
            "--no-nccl",
            "--allow-cuda-mismatch",
            "--with-ffmpeg",
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("master", result.stdout)
        self.assertFalse(
            any(
                command[0] in ("curl", "cargo", "apt-get")
                for command in self.commands()
            )
        )
        self.assert_no_installation()

    def test_latest_source_build_resolves_a_stable_tag(self):
        self.env["INSTALL_TEST_ALLOW_BUILD"] = "1"
        result = self.run_installer(
            "--yes", "--from-source", "--release", "latest", "--backend", "cpu"
        )
        self.assert_installed(result)
        builds = [command for command in self.commands() if command[0] == "cargo"]
        self.assertEqual(len(builds), 1)
        self.assertEqual(builds[0][builds[0].index("--tag") + 1], "v0.9.5")
        self.assertNotIn("--branch", builds[0])
        self.assertTrue(
            any(command[0] == "curl" and "-w" in command for command in self.commands())
        )

    def test_master_cpu_source_build_skips_cuda_prerequisites(self):
        self.env.update(INSTALL_TEST_ALLOW_BUILD="1", INSTALL_TEST_NVIDIA="1")
        result = self.run_installer("--yes", "--release", "master", "--backend", "cpu")
        self.assert_installed(result)
        builds = [command for command in self.commands() if command[0] == "cargo"]
        self.assertEqual(len(builds), 1)
        self.assertEqual(builds[0][builds[0].index("--branch") + 1], "master")
        self.assertNotIn("--tag", builds[0])
        self.assertNotIn("cuda", " ".join(builds[0]))
        self.assertFalse(
            any(command[0] in ("curl", "nvcc") for command in self.commands())
        )

    def test_macos_cpu_source_build_skips_metal_prerequisites(self):
        self.env.update(
            INSTALL_TEST_ALLOW_BUILD="1",
            INSTALL_TEST_OS="Darwin",
            INSTALL_TEST_ARCH="arm64",
        )
        result = self.run_installer("--yes", "--release", "v0.9.5", "--backend", "cpu")
        self.assert_installed(result)
        builds = [command for command in self.commands() if command[0] == "cargo"]
        self.assertEqual(len(builds), 1)
        self.assertNotIn("metal", " ".join(builds[0]))
        self.assertFalse(
            any(
                command[0] in ("xcrun", "xcodebuild", "xcode-select")
                for command in self.commands()
            )
        )

    def test_quiet_cuda_source_build_keeps_selected_backend_features(self):
        self.env.update(
            INSTALL_TEST_ALLOW_BUILD="1",
            INSTALL_TEST_NVIDIA="1",
            INSTALL_TEST_CUDA_TOOLKIT="1",
        )
        result = self.run_installer(
            "--yes",
            "--from-source",
            "--release",
            "master",
            "--backend",
            "cuda",
            "--no-nccl",
        )
        self.assert_installed(result)
        builds = [command for command in self.commands() if command[0] == "cargo"]
        self.assertEqual(len(builds), 1)
        features = set(builds[0][builds[0].index("--features") + 1].split())
        self.assertTrue({"cuda", "flash-attn"}.issubset(features), features)
        self.assertFalse({"nccl", "metal"} & features, features)
        self.assertRegex(result.stdout, r"Acceleration\s+NVIDIA CUDA")
        self.assertNotIn("info:", result.stdout)
        self.assertNotRegex(
            result.stdout, r"(?i)(enabling|skipping) (cuda|nccl|cudnn|flash)"
        )

    def test_installation_choices_persist_on_rerun(self):
        self.env.update(INSTALL_TEST_ALLOW_BUILD="1", INSTALL_TEST_NVIDIA="1")
        result = self.run_installer(
            "--yes",
            "--from-source",
            "--release",
            "v0.9.5",
            "--backend",
            "cpu",
            "--no-nccl",
            "--allow-cuda-mismatch",
        )
        self.assert_installed(result)
        config = (self.home / ".mistralrs/install.conf").read_text()
        for line in (
            "release=v0.9.5",
            "backend=cpu",
            "method=source",
            "no_nccl=1",
            "allow_cuda_mismatch=1",
        ):
            self.assertIn(line + "\n", config)
        result = self.run_installer("--yes")
        self.assert_installed(result)
        builds = [command for command in self.commands() if command[0] == "cargo"]
        self.assertEqual(len(builds), 2)
        self.assertEqual(builds[-1][builds[-1].index("--tag") + 1], "v0.9.5")
        self.assertNotIn("cuda", " ".join(builds[-1]))
        self.assertEqual((self.home / ".mistralrs/install.conf").read_text(), config)
        self.assertEqual(
            (self.home / ".profile").read_text().count('. "$HOME/.mistralrs/env"'), 1
        )

    def test_updater_environment_and_cli_override_saved_release(self):
        config = self.home / ".mistralrs/install.conf"
        config.parent.mkdir()
        config.write_text("release=v0.9.4\nbackend=cpu\nmethod=auto\n")
        self.env["MISTRALRS_INSTALL_TAG"] = "v0.9.5"
        result = self.run_installer("--yes")
        self.assert_installed(result)
        result = self.run_installer("--yes", "--release", "v0.9.6")
        self.assert_installed(result)
        self.assertEqual(
            self.download_urls(),
            [
                f"{RELEASES_URL}/download/v0.9.5/{CPU_ASSET}",
                f"{RELEASES_URL}/download/v0.9.6/{CPU_ASSET}",
            ],
        )
        self.assertIn("release=v0.9.6\n", config.read_text())

    def test_interactive_customization_changes_release_and_backend(self):
        self.env["INSTALL_TEST_NVIDIA"] = "1"
        result = self.run_interactive(
            [
                (MENU_READY, "c"),
                (MENU_READY, "1"),
                (MENU_READY, "t"),
                (r"(?i)tag[^:\n]*:", "v0.9.5\n"),
                (MENU_READY, "2"),
                (MENU_READY, "c"),
                (MENU_READY, "b"),
                (MENU_READY, "i"),
            ],
            "--dry-run",
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertRegex(result.stdout, r"Version\s+v0\.9\.5")
        self.assertRegex(result.stdout, r"Acceleration\s+CPU")
        self.assertFalse(
            any(
                command[0] in ("curl", "cargo", "apt-get")
                for command in self.commands()
            )
        )
        self.assert_no_installation()

    def test_interactive_cancel_does_not_install(self):
        result = self.run_interactive([(MENU_READY, "q")])
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("cancelled", result.stdout.lower())
        self.assertFalse(
            any(
                command[0] in ("curl", "cargo", "apt-get")
                for command in self.commands()
            )
        )
        self.assert_no_installation()

    def test_piped_script_reads_customization_from_terminal(self):
        result = self.run_interactive(
            [
                (MENU_READY, "c"),
                (MENU_READY, "1"),
                (MENU_READY, "t"),
                (r"(?i)tag[^:\n]*:", "v0.9.5\n"),
                (MENU_READY, "b"),
                (MENU_READY, "i"),
            ],
            "--dry-run",
            piped=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertRegex(result.stdout, r"Version\s+v0\.9\.5")
        self.assertIn("Preview complete", result.stdout)
        self.assertFalse(
            any(
                command[0] in ("curl", "cargo", "apt-get")
                for command in self.commands()
            )
        )
        self.assert_no_installation()

    def test_arrow_selection_enter_and_escape_work_without_line_input(self):
        result = self.run_interactive(
            [
                (MENU_READY, "\x1b[B"),
                (r"(?s)>\s*\[c\]\s*Customize.*" + MENU_READY, "\r"),
                (MENU_READY, "\x1b"),
                (MENU_READY, "\x1b[A"),
                (r"(?s)>\s*\[q\]\s*Cancel.*" + MENU_READY, "\r"),
            ],
            "--dry-run",
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("cancelled", result.stdout.lower())
        self.assert_no_installation()

    def test_escape_cancels_main_menu(self):
        result = self.run_interactive([(MENU_READY, "\x1b")])
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("cancelled", result.stdout.lower())
        self.assert_no_installation()

    def test_control_c_restores_terminal_and_cursor(self):
        result = self.run_interactive([(MENU_READY, "\x03")])
        self.assertEqual(result.returncode, 130, result.stdout)
        self.assertIn("\x1b[?25l", result.raw_stdout)
        self.assertGreater(
            result.raw_stdout.rfind("\x1b[?25h"), result.raw_stdout.rfind("\x1b[?25l")
        )
        self.assert_no_installation()

    def test_rich_terminal_displays_logo_and_colors(self):
        result = self.run_interactive([(MENU_READY, "q")])
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("__", result.stdout)
        logo = result.raw_stdout.split("Install mistral.rs", 1)[0]
        self.assertIn("\x1b[1m", logo)
        self.assertNotRegex(logo, r"\x1b\[(?:[0-9]+;)*(?:3[0-8]|9[0-7])(?:;[0-9]+)*m")
        menu = result.raw_stdout.split("\x1b[?25l", 1)[1]
        self.assertRegex(menu, r"\x1b\[[0-9;]*3[0-7][;m]")
        self.assert_no_installation()

    def test_no_color_keeps_single_key_menu_navigation(self):
        result = self.run_interactive([(MENU_READY, "q")], no_color=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotRegex(result.raw_stdout, r"\x1b\[[0-9;]*m")
        self.assert_no_installation()

    def test_repeated_navigation_preserves_logo_and_installation_plan(self):
        result = self.run_interactive(
            [(MENU_READY, "\x1b[B")] * 6 + [(MENU_READY, "q")], "--dry-run"
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        first_menu = result.raw_stdout.index("\x1b[?25l")
        original_plan = self.terminal_screen(result.raw_stdout[:first_menu]).rstrip(
            "\n"
        )
        final_screen = self.terminal_screen(result.raw_stdout)
        self.assertTrue(final_screen.startswith(original_plan), final_screen)
        self.assert_no_installation()

    def test_unattended_install_requires_yes_before_changes(self):
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("--yes", result.stdout)
        self.assertFalse(
            any(
                command[0] in ("curl", "cargo", "apt-get")
                for command in self.commands()
            )
        )
        self.assert_no_installation()

    def test_success_ends_with_browser_chat_instructions(self):
        result = self.run_installer("--yes", "--backend", "cpu")
        self.assert_installed(result)
        command = re.search(r"mistralrs serve[^\n]*", result.stdout)
        self.assertIsNotNone(command, result.stdout)
        self.assertIn("-m Qwen/Qwen3-4B", command.group())
        self.assertIn("--quant 4", command.group())
        self.assertIn("http://localhost:1234/ui", result.stdout)
        self.assertNotIn("mistralrs run", result.stdout)
        self.assertNotIn("--agent", result.stdout)
        for unwanted in (
            "start now",
            "[ finish ]",
            "downloads on first run",
            "downloads the model on first run",
        ):
            self.assertNotIn(unwanted, result.stdout.lower())
        self.assertFalse(
            any(command[0] in ("sudo", "apt-get") for command in self.commands())
        )

    def test_redirected_prebuilt_progress_is_ordered_and_static(self):
        self.env["TERM"] = "xterm-256color"
        self.env.pop("NO_COLOR")
        result = self.run_installer("--yes", "--backend", "cpu")
        self.assert_installed(result)
        self.assert_stage_order(result, "Download binary")
        self.assertNotIn("\x1b", result.raw_stdout)
        self.assertNotIn("\r", result.raw_stdout)
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_source_progress_is_ordered_without_a_fake_percentage(self):
        self.env["INSTALL_TEST_ALLOW_BUILD"] = "1"
        result = self.run_installer(
            "--yes", "--from-source", "--release", "master", "--backend", "cpu"
        )
        self.assert_installed(result)
        self.assert_stage_order(result, "Build from source")
        self.assertNotRegex(result.stdout, r"[0-9]+%")
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_live_download_progress_reports_actual_bytes_and_percentage(self):
        self.env["INSTALL_TEST_DOWNLOAD_DELAY"] = str(PROGRESS_SAMPLE_SECONDS)
        result = self.run_interactive([], "--yes", "--backend", "cpu")
        self.assert_installed(result)
        percentages = re.findall(
            r"([0-9]+)%\s+[0-9.]+\s+\w+\s+/\s+[0-9.]+\s+\w+", result.stdout
        )
        self.assertTrue(
            any(0 < int(value) < 100 for value in percentages), result.stdout
        )
        self.assert_stage_order(result, "Download binary")
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_live_cargo_progress_uses_native_completed_unit_count(self):
        self.env.update(
            INSTALL_TEST_ALLOW_BUILD="1",
            INSTALL_TEST_BUILD_COUNTERS="1",
            INSTALL_TEST_COUNTER_DELAY=str(PROGRESS_SAMPLE_SECONDS),
        )
        result = self.run_interactive(
            [], "--yes", "--from-source", "--release", "master", "--backend", "cpu"
        )
        self.assert_installed(result)
        self.assertRegex(result.stdout, r"\b8/20\b")
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_live_cargo_counters_with_utf8_lc_all(self):
        self.check_utf8_cargo_counters(lang_only=False)

    def test_live_cargo_counters_with_utf8_lang_only(self):
        self.check_utf8_cargo_counters(lang_only=True)

    def test_full_unicode_failure_with_utf8_lc_all(self):
        self.check_utf8_failure_diagnostics(lang_only=False)

    def test_full_unicode_failure_with_utf8_lang_only(self):
        self.check_utf8_failure_diagnostics(lang_only=True)

    def test_utf8_menu_ignores_unknown_unicode_key_and_restores_terminal(self):
        self.use_utf8_locale(lang_only=True)
        result = self.run_interactive([(MENU_READY, "\u00e9"), (MENU_READY, "q")])
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("cancelled", result.stdout.lower())
        self.assertNotIn("Illegal byte sequence", result.stdout)
        self.assert_no_installation()

    def test_verbose_streams_stdout_and_stderr_before_command_finishes(self):
        existing = self.home / ".mistralrs" / "mistralrs"
        existing.parent.mkdir()
        existing.write_text("existing install\n")
        self.env.update(
            INSTALL_TEST_ALLOW_BUILD="1",
            INSTALL_TEST_VERBOSE_LINES="1",
            INSTALL_TEST_BUILD_DELAY=str(LONG_RUNNING_COMMAND_SECONDS),
        )
        result = self.run_interactive(
            [
                (MENU_READY, "i"),
                (
                    r"(?s)fixture stdout live marker.*fixture stderr live marker",
                    "\x03",
                    True,
                ),
            ],
            "--verbose",
            "--from-source",
            "--release",
            "master",
            "--backend",
            "cpu",
        )
        self.assertEqual(result.returncode, 130, result.stdout)
        self.assertEqual(existing.read_text(), "existing install\n")
        command_pid = int(self.command_pid.read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(command_pid, 0)
        log = self.assert_diagnostic_log(result, "fixture stdout live marker")
        self.assertIn("fixture stderr live marker", log.read_text())

    def test_verbose_success_streams_output_once_and_removes_log(self):
        self.env.update(INSTALL_TEST_ALLOW_BUILD="1", INSTALL_TEST_VERBOSE_LINES="1")
        result = self.run_installer(
            "--yes",
            "--verbose",
            "--from-source",
            "--release",
            "master",
            "--backend",
            "cpu",
        )
        self.assert_installed(result)
        self.assert_stage_order(result, "Build from source")
        self.assertEqual(result.stdout.count("fixture stdout live marker"), 1)
        self.assertEqual(result.stdout.count("fixture stderr live marker"), 1)
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_script_edit_during_build_does_not_read_modified_tail(self):
        result = self.run_installer_with_edit_during_build(INSTALLER.read_text())
        self.assert_installed(result)
        self.assertIn("installed successfully", result.stdout)
        self.assertNotIn("unexpected EOF", result.stdout)
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_live_build_progress_can_be_cancelled_and_stops_the_command(self):
        existing = self.home / ".mistralrs" / "mistralrs"
        existing.parent.mkdir()
        existing.write_text("existing install\n")
        self.env.update(
            INSTALL_TEST_ALLOW_BUILD="1",
            INSTALL_TEST_BUILD_DELAY=str(LONG_RUNNING_COMMAND_SECONDS),
        )
        result = self.run_interactive(
            [
                (MENU_READY, "i"),
                ("Compiling installer-fixture v0.9.5", "\x03", True),
            ],
            "--from-source",
            "--release",
            "master",
            "--backend",
            "cpu",
        )
        self.assertEqual(result.returncode, 130, result.stdout)
        self.assertEqual(existing.read_text(), "existing install\n")
        self.assertNotRegex(result.stdout, r"[0-9]+%")
        self.assertRegex(
            result.stdout,
            r"\[[|/\\-]\]\s+[0-9]+m [0-9]{2}s\s+Compiling installer-fixture",
        )
        self.assertGreater(
            result.raw_stdout.rfind("\x1b[?25h"), result.raw_stdout.rfind("\x1b[?25l")
        )
        command_pid = int(self.command_pid.read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(command_pid, 0)
        self.assert_diagnostic_log(result, "Compiling installer-fixture v0.9.5")

    def test_optional_ffmpeg_failure_still_finishes_installation(self):
        result = self.run_installer("--yes", "--backend", "cpu", "--with-ffmpeg")
        self.assert_installed(result)
        self.assertTrue(
            any(command[0] == "apt-get" for command in self.commands()), result.stdout
        )
        self.assertIn("http://localhost:1234/ui", result.stdout)
        self.assertIn("ffmpeg", result.stdout.lower())

    def test_download_failure_preserves_existing_install_and_never_builds(self):
        existing = self.home / ".mistralrs" / "mistralrs"
        existing.parent.mkdir()
        existing.write_text("existing install\n")
        self.env["INSTALL_TEST_DOWNLOAD_FAIL"] = "1"
        result = self.run_installer("--yes", "--backend", "cpu")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(existing.read_text(), "existing install\n")
        self.assertFalse(any(command[0] == "cargo" for command in self.commands()))
        self.assertFalse((self.home / ".profile").exists())
        self.assert_diagnostic_log(result, "curl: simulated download failure")
        self.assertRegex(result.stdout, r"(?i)(retry|rerun|try again)")

    def test_build_failure_preserves_existing_install_and_reports_log(self):
        existing = self.home / ".mistralrs" / "mistralrs"
        existing.parent.mkdir()
        existing.write_text("existing install\n")
        self.env.update(INSTALL_TEST_ALLOW_BUILD="1", INSTALL_TEST_BUILD_FAIL="1")
        result = self.run_installer(
            "--yes", "--from-source", "--release", "v0.9.5", "--backend", "cpu"
        )
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(existing.read_text(), "existing install\n")
        self.assertFalse((self.home / ".profile").exists())
        self.assert_diagnostic_log(result, "error: simulated Cargo build failure")
        self.assertRegex(result.stdout, r"(?i)(retry|rerun|try again)")

    def test_failed_build_displays_complete_diagnostic_output(self):
        self.env.update(
            INSTALL_TEST_ALLOW_BUILD="1",
            INSTALL_TEST_BUILD_FAIL="1",
            INSTALL_TEST_DIAGNOSTIC_CONTEXT_LINES=str(DIAGNOSTIC_CONTEXT_LINES),
        )
        result = self.run_installer(
            "--yes", "--from-source", "--release", "master", "--backend", "cpu"
        )
        self.assertNotEqual(result.returncode, 0, result.stdout)
        for line in (
            "Compiling installer-fixture v0.9.5",
            "fixture stdout before failure",
            "error[E0425]: cannot find value fixture_symbol",
            " --> src/fixture.rs:7:9",
            "fixture diagnostic final context",
            "error: simulated Cargo build failure",
        ):
            self.assertIn(line, result.stdout)
        for index in range(DIAGNOSTIC_CONTEXT_LINES):
            self.assertIn(f"fixture diagnostic context {index:03d}", result.stdout)
        self.assertNotIn("fixture-progress", result.stdout)
        self.assertNotIn("\x1b", result.raw_stdout)
        self.assertNotIn("\r", result.raw_stdout)
        log = self.assert_diagnostic_log(result, "fixture diagnostic final context")
        self.assertIn(b"\x1b[31merror[E0425]", log.read_bytes())
        self.assertIn(b"3/20: fixture-progress", log.read_bytes())

    def test_verbose_build_failure_preserves_install_and_full_log(self):
        existing = self.home / ".mistralrs" / "mistralrs"
        existing.parent.mkdir()
        existing.write_text("existing install\n")
        self.env.update(
            INSTALL_TEST_ALLOW_BUILD="1",
            INSTALL_TEST_BUILD_FAIL="1",
            INSTALL_TEST_DIAGNOSTIC_CONTEXT_LINES=str(DIAGNOSTIC_CONTEXT_LINES),
        )
        result = self.run_installer(
            "--yes",
            "--verbose",
            "--from-source",
            "--release",
            "master",
            "--backend",
            "cpu",
        )
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(existing.read_text(), "existing install\n")
        self.assertIn("error[E0425]: cannot find value fixture_symbol", result.stdout)
        for index in range(DIAGNOSTIC_CONTEXT_LINES):
            self.assertEqual(
                result.stdout.count(f"fixture diagnostic context {index:03d}"), 1
            )
        self.assertIn("fixture diagnostic final context", result.stdout)
        self.assert_diagnostic_log(result, "fixture diagnostic final context")

    def test_unrunnable_prebuilt_preserves_existing_install(self):
        existing = self.home / ".mistralrs" / "mistralrs"
        existing.parent.mkdir()
        existing.write_text("existing install\n")
        self.write_archive(b"#!/bin/sh\nexit 1\n")
        result = self.run_installer("--yes", "--backend", "cpu")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(existing.read_text(), "existing install\n")
        self.assertFalse(any(command[0] == "cargo" for command in self.commands()))
        self.assertFalse((self.home / ".profile").exists())
        self.assert_diagnostic_log(result)

    def test_activation_failure_restores_existing_install(self):
        existing = self.home / ".mistralrs" / "mistralrs"
        existing.parent.mkdir()
        existing.write_text("existing install\n")
        self.write_archive(
            b'#!/bin/sh\n[ "$0" != "$HOME/.mistralrs/mistralrs" ] || exit 1\n'
            b"printf 'mistralrs 0.9.5\\n'\n"
        )
        result = self.run_installer("--yes", "--backend", "cpu")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(existing.read_text(), "existing install\n")
        self.assertFalse(any(command[0] == "cargo" for command in self.commands()))
        self.assertFalse((self.home / ".profile").exists())
        self.assert_diagnostic_log(result)
        self.assertEqual(list(self.home.glob(".mistralrs-*")), [])

    def test_interrupt_after_backup_restores_existing_install(self):
        existing = self.home / ".mistralrs" / "mistralrs"
        existing.parent.mkdir()
        existing.write_text("existing install\n")
        self.env["INSTALL_TEST_REAL_MV"] = shutil.which("mv")
        mv = self.bin / "mv"
        mv.unlink()
        mv.write_text(f"#!{sys.executable}\n" + textwrap.dedent(STUB_SCRIPT))
        mv.chmod(0o755)
        result = self.run_installer("--yes", "--backend", "cpu")
        self.assertEqual(result.returncode, 130, result.stdout)
        self.assertEqual(existing.read_text(), "existing install\n")
        self.assertEqual(list(self.home.glob(".mistralrs-*")), [])
        self.assertNotIn("installed successfully", result.stdout)
        self.assert_diagnostic_log(result)


if __name__ == "__main__":
    unittest.main()
