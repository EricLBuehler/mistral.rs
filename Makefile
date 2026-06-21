.PHONY: fmt docs-regen docs-check

fmt:
	cargo fmt
	ruff format
	find mistralrs-* -type f \( -name "*.metal" -o -name "*.c" -o -name "*.cu" -o -name "*.hpp" -o -name "*.h" -o -name "*.cpp" \) -exec clang-format -i {} +
docs-regen:
	cargo test -p mistralrs-cli regenerate_cli_reference -- --ignored
	cargo test -p mistralrs-server-core regenerate_openapi -- --ignored
	python3 docs/scripts/render_pyi.py
	python3 docs/scripts/render_examples.py

docs-check:
	cargo test -p mistralrs-cli docgen::cli_reference_matches_committed
	cargo test -p mistralrs-server-core openapi_matches_committed
	python3 docs/scripts/render_pyi.py --check
