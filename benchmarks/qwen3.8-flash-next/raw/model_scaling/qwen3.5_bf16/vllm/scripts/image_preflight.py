"""Inspect installed wheel metadata and tokenize prompts without importing vLLM or Torch."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

from tokenizers import Tokenizer

EXPECTED_VERSION = "0.28.0"
EXPECTED_COMMIT = "2cf0a6915ce544dc493a0990f2ea38d81601128a"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checkpoint = Path(sys.argv[1])
    prompts = json.loads(Path(sys.argv[2]).read_text())
    distribution = importlib.metadata.distribution("vllm")
    package = Path(distribution.locate_file("vllm"))
    version_file = package / "_version.py"
    assert distribution.version == EXPECTED_VERSION
    assert "g" + EXPECTED_COMMIT[:9] in version_file.read_text()
    protocol = package / "entrypoints/openai/engine/protocol.py"
    completion = package / "entrypoints/openai/completion/protocol.py"
    argument_file = package / "engine/arg_utils.py"
    argument_source = argument_file.read_text()
    assert (
        "--enable-log-requests" in argument_source
        and "BooleanOptionalAction" in argument_source
    )
    protocol_source = protocol.read_text()
    completion_source = completion.read_text()
    assert re.search(r"ConfigDict\([^)]*extra\s*=\s*[\"']allow[\"']", protocol_source)
    assert re.search(r"ignore_eos\s*:\s*bool", completion_source)
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    rows = []
    for name, prompt in prompts.items():
        plain = tokenizer.encode(prompt, add_special_tokens=False).ids
        special = tokenizer.encode(prompt, add_special_tokens=True).ids
        assert plain == special, name
        rows.append(
            {
                "name": name,
                "prompt_tokens": len(plain),
                "token_ids": plain,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "special_tokens_true_false_equal": True,
            }
        )
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "vllm",
            "torch",
            "triton",
            "transformers",
            "tokenizers",
            "flashinfer-python",
        )
    }
    files = [
        {"path": str(path), "sha256": sha(path)}
        for path in (version_file, protocol, completion, argument_file)
    ]
    print(
        json.dumps(
            {
                "complete": True,
                "versions": versions,
                "vllm_commit_prefix": "g" + EXPECTED_COMMIT[:9],
                "config_sha256": sha(checkpoint / "config.json"),
                "tokenizer_sha256": sha(checkpoint / "tokenizer.json"),
                "prompts": rows,
                "source_files": files,
                "cache_prompt": "Accepted as an extra field but unused; --no-enable-prefix-caching is required.",
                "scope": "Package files and tokenizers only; vLLM/Torch were not imported, no GPU requested.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
