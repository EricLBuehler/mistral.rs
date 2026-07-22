"""Run the complete dynamic LoRA lifecycle on a small public model.

> This example targets the current source API. Published v0.9.0 packages do not
> include dynamic LoRA; use a [current source build](/mistral.rs/developer/from-source/)
> until the next release.

Run with:

~~~bash
python examples/python/lora.py
~~~

The example downloads two pinned adapters, loads one into an initially empty
runtime, routes base, alias, and exact-generation requests, replaces the alias,
checks stale-CAS rejection, and unloads it.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

from mistralrs import ChatCompletionRequest, LoraAdapterError, Runner, Which


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_REPO = "closestfriend/brie-qwen2.5-0.5b"
ADAPTER_REVISION = "acad7d767bece1486f2e6644820f784b3bcb6b5e"
REPLACEMENT_REPO = "axel-datos/qwen2.5-0.5b-instruct_MATH_qlora"
REPLACEMENT_REVISION = "0fd79bf6bfcc41446ccbb38e3304ea079af034ab"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def download_adapter(repo: str, revision: str, directory: Path) -> None:
    for filename in ADAPTER_FILES:
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
        urlretrieve(url, directory / filename)


def generate(runner: Runner, label: str, adapter=None):
    response = runner.send_chat_completion_request(
        ChatCompletionRequest(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Explain why checking intermediate steps is useful "
                        "when solving a difficult problem."
                    ),
                }
            ],
            model="default",
            max_tokens=160,
            temperature=0.0,
            adapter=adapter,
        )
    )
    print(f"\n[{label}; generation={response.adapter_generation}]")
    print(response.choices[0].message.content)
    return response


runner = Runner(which=Which.Lora(model_id=BASE_MODEL))

with TemporaryDirectory(prefix="mistralrs-lora-") as directory:
    adapter_dir = Path(directory) / "initial"
    replacement_dir = Path(directory) / "replacement"
    adapter_dir.mkdir()
    replacement_dir.mkdir()
    download_adapter(ADAPTER_REPO, ADAPTER_REVISION, adapter_dir)
    download_adapter(REPLACEMENT_REPO, REPLACEMENT_REVISION, replacement_dir)

    loaded = runner.load_lora_adapter("production", adapter_dir)
    status = runner.lora_adapter_status()
    assert len(status.adapters) == 1
    assert status.adapters[0].generation == loaded.generation
    print(f"Loaded {loaded.alias} as generation {loaded.generation}")
    print(f"Resident adapter bytes: {status.resident_bytes}")

    base = generate(runner, "base")
    alias = generate(runner, "alias", loaded.alias)
    exact = generate(runner, "exact generation", loaded.exact())
    assert base.adapter_generation is None
    assert alias.adapter_generation == loaded.generation
    assert exact.adapter_generation == loaded.generation

    replaced = runner.load_lora_adapter(
        loaded.alias,
        replacement_dir,
        load_inplace=True,
        expected_generation=loaded.generation,
    )
    assert replaced.generation != loaded.generation
    replacement = generate(runner, "replacement", replaced.alias)
    assert replacement.adapter_generation == replaced.generation

    try:
        runner.load_lora_adapter(
            loaded.alias,
            adapter_dir,
            load_inplace=True,
            expected_generation=loaded.generation,
        )
    except LoraAdapterError as error:
        assert error.code == "lora_generation_mismatch"
    else:
        raise AssertionError("stale generation unexpectedly replaced the adapter")

    unloaded = runner.unload_lora_adapter(
        loaded.alias,
        expected_generation=replaced.generation,
    )
    assert unloaded.generation == replaced.generation
    assert runner.list_lora_adapters() == []
    print(f"Unloaded {unloaded.alias} with generation CAS")
