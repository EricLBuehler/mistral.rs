from mistralrs import (
    Runner,
    Which,
    DiffusionArchitecture,
    ImageGenerationResponseFormat,
)

runner = Runner(
    which=Which.DiffusionPlain(
        model_id="black-forest-labs/FLUX.1-schnell",
        arch=DiffusionArchitecture.FluxOffloaded,
    ),
)

res = runner.generate_image(
    "A vibrant sunset in the mountains, 4k, high quality.",
    ImageGenerationResponseFormat.Url,
)
print(res.data[0].url)
