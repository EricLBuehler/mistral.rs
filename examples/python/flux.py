from mistralrs import (
    Runner,
    Which,
    DiffusionArchitecture,
    ImageGenerationResponseFormat,
)

runner = Runner(
    which=Which.DiffusionPlain(
        model_id="mistralai/Mistral-7B-Instruct-v0.1",
        arch=DiffusionArchitecture.FluxOffloaded,
    ),
)

res = runner.generate_image(
    "A vibrant sunset in the mountains, 4k, high quality.",
    ImageGenerationResponseFormat.Url,
)
print(res.choices[0].url)
