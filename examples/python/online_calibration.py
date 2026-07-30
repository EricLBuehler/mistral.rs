from mistralrs import Runner, Which, ChatCompletionRequest

runner = Runner(
    which=Which.Plain(
        model_id="google/gemma-4-E4B-it",
    ),
    in_situ_quant="Q4K",
)

request = ChatCompletionRequest(
    model="default",
    messages=[{"role": "user", "content": "Explain how a hash map works, briefly."}],
    max_tokens=64,
)

# Collect activation statistics while serving normally (~15% decode overhead while on).
runner.begin_calibration()
for _ in range(8):
    runner.send_chat_completion_request(request)

status = runner.calibration_status()
print(
    f"Collecting on {status.layers_tracking}/{status.layers} layers, "
    f"{status.total_rows} token rows seen"
)

# Requantize from the source weights with the traffic-derived importance matrix and
# hot-swap each layer. The optional path also saves the imatrix for reuse.
runner.apply_calibration(save_cimatrix="traffic.cimatrix")

res = runner.send_chat_completion_request(request)
print(res.choices[0].message.content)
