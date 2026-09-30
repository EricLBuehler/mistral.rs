# Expert-kernel headroom controls

This followup separates padded arithmetic from selected-weight access costs. It does not change production kernels or replace the preceding serving benchmark.

- [Selected-weight scan](selected_scan/report.md): checksum-validated reads of exact packed expert bytes, with no model computation, bracketed by frozen complete-FFN replays. Native M56 medians are 2.466 ms scan and 3.403 ms FFN; all four chunk sizes and the separate cache-flush control are retained.
- [L2 refill profile](l2_refill/README.md): three warmed projections refill 628.07 MB against 610.71 MB of selected expert weights. These are L2 interface byte equivalents, not total LPDDR traffic or utilization.
- [Paired route accounting](accounting/paired_sequence_reuse.md): the exact sequences within each native B8xQ7 capture obtain a median 1.469x logical selected-weight reuse factor; B6xQ7 gives 1.384x.
- [Operation accounting](accounting/report.md): exact padding, grids, logical bytes, historical overhead, and a proposed integer-MMA design that has not been implemented or adopted.

The independent archives contain source, commands, validation, numerical diagnostics, raw samples, and hashes. Model weights and probe executables are excluded; their original paths and hashes remain recorded. Each subdirectory has its own checksum manifest. Timing and counter evidence cover one layer with fixed-Q7 diagnostic routes; neither the scan nor the profile establishes a full-model bandwidth ceiling or an attainable optimization budget.

The current committed source passes the release CLI [cargo check](cargo_check.log), with its [command and log hash](validation.json). GPU work and compilation finished before the six paused editor processes were restored, using identity checks recorded in [editor_restoration.json](editor_restoration.json).
