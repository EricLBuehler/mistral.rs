#!/usr/bin/env python3
import json
from pathlib import Path
import subprocess

work = Path(__file__).resolve().parent
manifest = json.loads((work / "manifest.json").read_text())
results = {}
for variant in ("red", "green"):
    binary = work / f"test_{variant}"
    command = [
        "rustc", "--edition=2021", "--test", "-C", "target-cpu=native",
        "--crate-name", f"autotuner_{variant}",
        "--extern", f"tracing={manifest['tracing_rlib']}",
        "-L", f"dependency={manifest['dependency_dir']}",
        str(work / f"wrapper_{variant}.rs"), "-o", str(binary),
    ]
    compiled = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (work / f"{variant}.compile.log").write_text(compiled.stdout)
    if compiled.returncode:
        print(compiled.stdout)
        raise SystemExit(f"{variant} failed to compile; this does not establish a test failure")
    command = [str(binary), manifest["test"], "--exact", "--nocapture"]
    tested = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (work / f"{variant}.run.log").write_text(tested.stdout)
    results[variant] = {"returncode": tested.returncode, "command": command}
    print(f"{variant}: return code {tested.returncode}")
    print(tested.stdout)
    if "running 1 test" not in tested.stdout:
        raise SystemExit(f"{variant} did not execute exactly one test")
(work / "results.json").write_text(json.dumps(results, indent=2) + "\n")
if results["red"]["returncode"] != 101 or results["green"]["returncode"] != 0:
    raise SystemExit("Expected the old exploration to fail and the new exploration to pass")
print("Verified: old exploration fails and new exploration passes the identical regression.")
