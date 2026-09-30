# Build the release data and figures

Run from a checkout containing the committed benchmark archives. Python 3.12.3 and the versions in `requirements.txt` produced the checked-in figures:

```bash
python3 -m pip install -r releases/v0.9.5/scripts/requirements.txt
python3 releases/v0.9.5/scripts/build_release.py
```

The generator reads repository-relative source files, excludes warmups, recomputes every plotted trial rate, and checks samples, means, and sample standard deviations against the existing validated summaries. It does not run a server or benchmark. `--repo-root` selects the input checkout; `--output-dir` writes another release output directory for comparison.

Outputs:

- `raw/summary.json`: 33 cells with protocol, units, mean, sample SD, five samples, source path, and secondary closed-loop metrics.
- `raw/summary.csv`: the same primary cells in a flat table.
- `raw/results.jsonl`: 165 measured trial rows, each linked to its source file and JSON pointer, with wall time and token/request counts.
- `raw/tables.md`: report-ready tables; the eight-prompt arithmetic mean has no invented repetition uncertainty.
- `raw/run_manifest.json`: source and output hashes, original run provenance, frozen binary identities, validation checks, and rendering-library versions.
- `figures/*.png` and `*.svg`: three static charts with zero-based axes and five-trial sample-SD error bars.

The figure families retain distinct prefill, finite-burst, and replenished closed-loop definitions. Homogeneous finite-wave controls and profiled timings are excluded. Full requests, logs, model provenance, and historical runs stay in their existing benchmark archives.

JSON, CSV, tables, PNG, and SVG outputs are deterministic in the recorded rendering environment. The SVG timestamp is omitted and element IDs use a fixed hash salt. Other library or font builds can render different bytes; the manifest records those versions rather than promising cross-platform pixel identity.
