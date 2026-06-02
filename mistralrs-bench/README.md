# `mistralrs-bench`

> **Deprecated:** The standalone `mistralrs-bench` package is deprecated. Use `mistralrs bench` instead for the same functionality.
>
> **Migration:**
> ```bash
> # Old
> cargo run --release --features cuda --package mistralrs-bench -- plain -m model-id
>
> # New
> mistralrs bench -m model-id
> ```

For installation, supported accelerators, and the full list of benchmark options, see the [top-level README](../README.md) and run `mistralrs bench --help`.