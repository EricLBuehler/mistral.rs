.PHONY: fmt

fmt:
	cargo fmt
	ruff format
	find mistralrs-* -type f \( -name "*.metal" -o -name "*.c" -o -name "*.cu" -o -name "*.hpp" -o -name "*.h" -o -name "*.cpp" \) -exec clang-format -i {} +