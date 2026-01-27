from mistralrs import (
    Runner,
    Which,
    ChatCompletionRequest,
    VisionArchitecture,
    VisionAutoMapParams,
)

# MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
MODEL_ID = "lamm-mit/Cephalo-Llama-3.2-11B-Vision-Instruct-128k"

runner = Runner(
    which=Which.VisionPlain(
        model_id=MODEL_ID,
        arch=VisionArchitecture.VLlama,
        auto_map_params=VisionAutoMapParams(
            max_seq_len=4096, max_batch_size=2, max_num_images=2, max_image_length=512
        ),
    ),
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://www.nhmagazine.com/content/uploads/2019/05/mtwashingtonFranconia-2-19-18-108-Edit-Edit.jpg"
                        },
                    },
                    {
                        "type": "text",
                        "text": "What is shown in this image? Write a detailed response analyzing the scene.",
                    },
                ],
            }
        ],
        max_tokens=256,
        presence_penalty=1.0,
        top_p=0.1,
        temperature=0.1,
    )
)
print(res.choices[0].message.content)
print(res.usage)
