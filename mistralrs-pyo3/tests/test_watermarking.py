"""Run with an installed mistralrs extension: python -m unittest discover -s mistralrs-pyo3/tests."""

import json
from pathlib import Path
import unittest

from mistralrs import (
    ChatCompletionRequest,
    CompletionRequest,
    SynthIdTextWatermarkConfig,
    WatermarkConfig,
)

CONFIGS = Path(__file__).resolve().parents[2] / "examples" / "watermarking"


class WatermarkBindings(unittest.TestCase):
    def test_every_token_scheme_accepts_requests_and_detects(self):
        for path in sorted(CONFIGS.glob("*.json")):
            config = json.loads(path.read_text())
            if config["scheme"] == "semstamp":
                continue
            with self.subTest(scheme=config["scheme"]):
                if "vocab_size" in config:
                    config["vocab_size"] = 8
                mark = WatermarkConfig(**config)
                self.assertEqual(mark.scheme, config["scheme"])
                ChatCompletionRequest(model="test", messages="hello", watermark=mark)
                CompletionRequest(model="test", prompt="hello", watermark=mark)
                evidence = mark.detect(
                    [1, 2, 3, 4, 5, 6], prompt_len=4, eos_token_ids=[6]
                )
                self.assertEqual(evidence.get("tokens_scored", evidence.get("trials")), 1)

    def test_semstamp_embeddings_and_invalid_configuration(self):
        config = json.loads((CONFIGS / "semstamp.json").read_text())
        mark = WatermarkConfig(**config)
        evidence = mark.detect_embeddings(
            [[1.0, 0.3, 0.5], [-0.2, 0.9, 0.1]], prompt_len=1
        )
        self.assertEqual(evidence["kind"], "semstamp")
        self.assertEqual(evidence["sentences_scored"], 1)
        self.assertIsInstance(
            mark.accepts_embedding([1.0, 0.3, 0.5], [-0.2, 0.9, 0.1]), bool
        )
        with self.assertRaises(ValueError):
            mark.detect([1, 2, 3])
        invalid_parameters = (
            {"scheme": "unknown"},
            {"scheme": "kgw"},
            {"scheme": "mpac", "vocab_size": 8, "payload": [2]},
            {"scheme": "synthid", "typo": 1},
        )
        for params in invalid_parameters:
            with self.subTest(params=params), self.assertRaises(ValueError):
                WatermarkConfig("00" * 32, **params)

    def test_existing_synthid_binding_remains_accepted(self):
        mark = SynthIdTextWatermarkConfig("00" * 32)
        ChatCompletionRequest(model="test", messages="hello", watermark=mark)
        CompletionRequest(model="test", prompt="hello", watermark=mark)
        self.assertEqual(mark.detect([1, 2, 3, 4, 5])[1], 1)


if __name__ == "__main__":
    unittest.main()
