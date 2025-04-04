from transformers import AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig
import shutil
import torch
import os
import argparse
from datasets import load_dataset


def get_wikitext2(tokenizer, nsamples, seqlen):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(
        lambda x: len(x["text"]) >= seqlen
    )

    return [tokenizer(example["text"]) for example in traindata.select(range(nsamples))]


@torch.no_grad()
def calculate_avg_ppl(model, tokenizer):
    from gptqmodel.utils import Perplexity

    ppl = Perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_path="wikitext",
        dataset_name="wikitext-2-raw-v1",
        split="train",
        text_column="text",
    )

    all = ppl.calculate(n_ctx=512, n_batch=512)

    # average ppl
    avg = sum(all) / len(all)

    return avg


def main(args):
    pretrained_model_dir = args.src
    quantized_model_dir = args.dst
    quantize_config = QuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=False,  # Set to False can significantly speed up inference but maeks the perplexity slightly worse
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
    if pretrained_model_dir[-1] != "/":
        pretrained_model_dir += "/"
    if quantized_model_dir[-1] != "/":
        quantized_model_dir += "/"

    # Load un-quantized model, by default, the model will always be loaded into CPU memory
    model = GPTQModel.load(pretrained_model_dir, quantize_config)
    model.resize_token_embeddings(len(tokenizer))
    traindataset = get_wikitext2(tokenizer, args.samples, args.seqlen)
    # Quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    model.quantize(traindataset)
    # Save quantized model
    model.save(quantized_model_dir)

    # Load quantized model, currently only support cpu or single gpu
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = GPTQModel.load(quantized_model_dir, device=device)

    # Inference with model.generate
    print(
        tokenizer.decode(
            model.generate(**tokenizer("test is", return_tensors="pt").to(device))[0]
        )
    )

    print(
        f"Quantized Model {quantized_model_dir} avg PPL is {calculate_avg_ppl(model, tokenizer)}"
    )
    shutil.copy2(pretrained_model_dir + "/tokenizer.json", quantized_model_dir)


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Transform uncompressed safetensors weights to 4-bit marlin-compatible format."
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[2, 3, 4, 8],
        help="Number of bits to use for quantization.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="Quantization group size. 128 offers good balance between speed/memory usage/quality. Use 32 for higher accuracy, lower speed/quality",
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Path to the directory where the safetensors and other configuration files are.",
    )
    parser.add_argument(
        "--dst",
        type=str,
        required=True,
        help="Path to save the transformed GPTQ model.",
    )

    parser.add_argument(
        "--samples", default=512, type=int, help="Number of samples for calibration."
    )
    parser.add_argument(
        "--seqlen",
        default=1024,
        type=int,
        help="Sample sequence length for calibration.",
    )

    args = parser.parse_args()
    if os.path.exists(args.src):
        if not os.path.exists(args.dst):
            os.makedirs(args.dst)
        main(args)
    else:
        print("Source folder not exists: ", args.src)
