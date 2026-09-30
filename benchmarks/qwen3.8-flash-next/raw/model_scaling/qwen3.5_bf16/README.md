# Same-checkpoint Qwen3.5 BF16 engine comparison

The comparison covers completed target-only mistral.rs and vLLM runs on one GB10, with the same local Qwen3.5-35B-A3B BF16 checkpoint and frozen request harness. Five measured trials follow two excluded warmups at C1, C6, and C8. See [comparison.md](comparison.md) and [comparison.json](comparison.json) for recomputed statistics, exact identities, protocol validation, operating ranges, and limitations.

The separate kernel_trace subtree establishes observed native cuTile MoE graph execution at C1 and C8. Its instrumented timings do not replace the unprofiled throughput results. Capture and analysis manifests retain the original hashes; kernel_proof.manifest.json additionally covers the later kernel proof, and graph_shape_proof.manifest.json covers the independent B8 shape proof.

The first vLLM attempt failed during a port preflight before Docker or GPU launch. It is preserved under failed_attempts and excluded from statistics. The successful vLLM source directory is vllm_bf16_retry1.

Tokenizer copies, compiler caches, large profiler reports, SQLite databases, model weights, and executable binaries remain external. archive_provenance.json records source paths and hashes for excluded run/trace artifacts; model, binary, and image identities also remain in the run metadata. Three unreadable container-owned cache files retain paths/sizes and explicit unavailable hashes. No model weights were copied or rehashed by this archive step. Potential secret fields are redacted from copies with original and sanitized digests retained.

SHA256SUMS.json covers every archived file except itself. Source manifests refer to original run layouts, including externally retained files.
