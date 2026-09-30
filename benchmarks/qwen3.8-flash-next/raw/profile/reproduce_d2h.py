#!/usr/bin/env python3
"""Reproduce vocabulary-sized D2H counts from existing, read-only Nsight exports."""

import argparse
import json
from pathlib import Path
import sqlite3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = json.loads(
        Path(__file__).with_name("draft_sampling_transfer_evidence.json").read_text()
    )
    cases = []
    for path in args.sqlite:
        with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA cache_size=-2048")
            counts = dict(
                connection.execute(evidence["query"], evidence["query_parameters"]).fetchone()
            )
        cases.append({"source_sqlite": str(path.resolve()), **counts})
    result = {
        "query": evidence["query"],
        "query_parameters": evidence["query_parameters"],
        "cases": cases,
        "caveat": evidence["caveat"],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
