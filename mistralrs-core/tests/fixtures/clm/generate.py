"""Generate tiny Qwen3/CLM parity fixtures using the upstream CLM implementation.

Usage: python generate.py /path/to/Contrastive-LM/CLM [/path/to/CLM_v0.1-8B.pt]
Reference CLM commit: bb42c6c5bf914fd449bed2f6ca65be80602cb1f7
Dependencies: torch 2.14.1, transformers 5.18.0, tokenizers, numpy, safetensors.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import Qwen3Config, Qwen3Model


def module(name):
    source = Path(sys.argv[1]) / "src" / "clm" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, source)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


heads, schema = module("heads"), module("schema")
root = Path(__file__).parent
torch.manual_seed(42)
cfg = Qwen3Config(
    vocab_size=32,
    hidden_size=8,
    intermediate_size=16,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=4,
    max_position_embeddings=2048,
    rope_theta=1000000,
    rms_norm_eps=1e-6,
    bos_token_id=None,
    eos_token_id=None,
    pad_token_id=None,
)
cfg._attn_implementation = "eager"
encoder = Qwen3Model(cfg).eval()
encoder_cfg = cfg.to_dict()
encoder_cfg.update(
    architectures=["Qwen3ForCausalLM"],
    torch_dtype="float32",
    rope_theta=1000000,
    use_sliding_window=False,
    max_window_layers=1,
)
(root / "encoder" / "config.json").write_text(json.dumps(encoder_cfg, indent=2) + "\n")
save_file(
    {f"model.{k}": v.contiguous() for k, v in encoder.state_dict().items()},
    root / "encoder" / "model.safetensors",
)
vocab = {
    word: i
    for i, word in enumerate(
        [
            "[UNK]",
            "Customer",
            "invoice",
            "charged",
            "twice",
            "urgent",
            "billing",
            "technical",
            "Is",
            "this",
            "Which",
            "team",
            "How",
            "angry",
            "Calm",
            "Frustrated",
            "Very",
            "Yes",
            "No",
            "true",
            "false",
            "Charges",
            "invoices",
            "refunds",
            "Bugs",
            "outages",
            ":",
            "?",
            ".",
            "and",
            "customer",
            "phone",
        ]
    )
}
tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer.save(str(root / "encoder" / "tokenizer.json"))
head_cfg = {
    "width": 6,
    "depth": 3,
    "projection_dim": 4,
    "hidden_size": 8,
    "activation": "gelu",
    "layernorm": True,
    "residual": False,
    "model": "encoder",
}
kwargs = {
    "width": 6,
    "depth": 3,
    "proj": 4,
    "hidden": 8,
    "activation": "gelu",
    "layernorm": True,
}
state_head, action_head = heads.make_head(**kwargs), heads.make_head(**kwargs)
torch.save(
    {
        "cfg": head_cfg,
        "state_head": state_head.state_dict(),
        "action_head": action_head.state_dict(),
        "logit_scale": torch.tensor(2.0),
    },
    root / "heads.pt",
)
(root / "config.json").write_text(
    json.dumps(
        {
            "model_type": "clm",
            "base_model": "encoder",
            "encoder_pooling": "last-token",
            "embedding_dim": 8,
            "checkpoints": ["heads.pt"],
        },
        indent=2,
    )
    + "\n"
)
questions = {
    "urgent": {"type": "noul", "instructions": "Is this urgent?"},
    "team": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {
            "billing": "Charges invoices refunds",
            "technical": "Bugs and outages",
            "other": None,
        },
    },
    "anger": {
        "type": "score",
        "instructions": "How angry?",
        "criteria": ["Calm", "Frustrated", {"level": "Very angry"}],
    },
    "custom": {
        "type": "noul",
        "instructions": {"question": "Is this billing?"},
        "criteria": {"true": "invoice", "false": []},
    },
}
request = {
    "model": "default",
    "state": {"Customer": "invoice charged twice", "details": ["phone", True]},
    "questions": questions,
    "temperature": 0.7,
}
pairs = schema.build_pairs(request["state"], questions)
pair = heads.HeadPair("fixture", str(root / "heads.pt"), "cpu").ensure()
tokens = {}


def embed(texts):
    result = []
    for text in texts:
        ids = tokenizer.encode(text).ids
        tokens[text] = ids
        with torch.no_grad():
            raw = encoder(torch.tensor([ids])).last_hidden_state[:, -1, :].numpy()[0]
        result.append(raw / (np.linalg.norm(raw) + 1e-12))
    return np.stack(result)


answers = {}
for key, (state, keys, candidates) in pairs.items():
    zs = pair.project_states(embed([state]))
    za = pair.project_actions(embed(candidates))
    logits = (pair.scale * (za @ zs[0]) / request["temperature"]).tolist()
    answers[key] = schema.answer_from_logits(questions[key], keys, logits)
(root / "reference.json").write_text(
    json.dumps(
        {
            "request": request,
            "pairs": {
                k: {"state": v[0], "keys": v[1], "candidates": v[2]}
                for k, v in pairs.items()
            },
            "answers": answers,
            "input_tokens": sum(map(len, tokens.values())),
        },
        indent=2,
    )
    + "\n"
)

if len(sys.argv) > 2:
    pair = heads.HeadPair("published", sys.argv[2], "cpu").ensure()
    values = ((np.arange(4 * 4096, dtype=np.float32) % 97) - 48).reshape(4, 4096) / 48
    values = values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-12)
    zs, za = pair.project_states(values[:1]), pair.project_actions(values[1:])
    logits = (pair.scale * (za @ zs[0])).tolist()
    (root / "published-head-reference.json").write_text(
        json.dumps({"logits": logits}, indent=2) + "\n"
    )
