# convert awq models to marlin compatible format
# example: python3 examples/convert_awq_marlin.py --src /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/ --dst /home/Meta-Llama-3.1-8B-Instruct-AWQ-INT4-Marlin/ --bits 4

from typing import List
import torch
import numpy
from safetensors.torch import load_file, save_file
import argparse
import os
import shutil


def get_scale_perms():
    scale_perm: List[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: List[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def pack_cols(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    assert q_w.shape == (size_k, size_n)

    pack_factor = 32 // num_bits
    assert size_n % pack_factor == 0

    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(numpy.uint32)

    q_res = numpy.zeros((size_k, size_n // pack_factor), dtype=numpy.uint32)

    for i in range(pack_factor):
        q_res |= q_w[:, i::pack_factor] << num_bits * i

    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    q_res = q_res.contiguous()

    return q_res


def unpack_cols(
    packed_q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    pack_factor = 32 // num_bits
    assert size_n % pack_factor == 0
    assert packed_q_w.shape == (size_k, size_n // pack_factor), (
        "packed_q_w.shape = {} size_k = {}, size_n = {} pack_Factor = {}".format(
            packed_q_w.shape, size_k, size_n, pack_factor
        )
    )

    orig_device = packed_q_w.device

    packed_q_w_cpu = packed_q_w.cpu().numpy().astype(numpy.uint32)
    q_res = numpy.zeros((size_k, size_n), dtype=numpy.uint32)

    mask = (1 << num_bits) - 1
    for i in range(pack_factor):
        vals = packed_q_w_cpu & mask
        packed_q_w_cpu >>= num_bits
        q_res[:, i::pack_factor] = vals

    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    q_res = q_res.contiguous()

    return q_res


def marlin_zero_points(
    zp: torch.Tensor, size_k: int, size_n: int, num_bits: int
) -> torch.Tensor:
    # Permute zero-points in a similar way to scales, but do not use the
    # "single" permutation, since zero-points are applied on every MMA
    scale_perm, _ = get_scale_perms()
    zp = zp.reshape((-1, len(scale_perm)))[:, scale_perm]

    # Interleave column dim (for the dequantize code) and pack it to int32
    if num_bits == 4:
        interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = numpy.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    zp = zp.reshape((-1, len(interleave)))[:, interleave].ravel()
    zp = zp.reshape((-1, size_n)).contiguous()
    zp = pack_cols(zp, num_bits, size_k, size_n)

    return zp


def awq_to_marlin_zero_points(
    q_zp_packed: torch.Tensor, size_k: int, size_n: int, num_bits: int
) -> torch.Tensor:
    # AWQ zero-points are quantized and packed on the column dim.
    # In addition, the values are permuted based on dequantizer.
    # Here we undo both of these, and then apply marlin permutation
    # and pack it back.
    q_zp = unpack_cols(q_zp_packed, num_bits, size_k, size_n)

    # Undo interleaving (use argsort(..) to get inverse perm)
    if num_bits == 4:
        undo_interleave = numpy.argsort(numpy.array([0, 2, 4, 6, 1, 3, 5, 7]))
    elif num_bits == 8:
        undo_interleave = numpy.argsort(numpy.array([0, 2, 1, 3]))
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    q_zp = q_zp.reshape((-1, len(undo_interleave)))[:, undo_interleave].ravel()
    q_zp = q_zp.reshape((-1, size_n)).contiguous()

    marlin_zp = marlin_zero_points(q_zp, size_k, size_n, num_bits)
    return marlin_zp


def transform_file(src_folder, dst_folder, bits):
    """
    Transform and save safetensors file.

    Args:
        src_folder (str): Path to the source safetensors file.
        dst_folder (str): Path to the target safetensors file.
    """
    if not os.path.exists(src_folder):
        raise FileNotFoundError(f"Source file not found: {src_folder}")
    print(f"Loading source file: {src_folder}")
    tgt_dict = {}

    files = os.listdir(src_folder)
    files = [k for k in files if k.endswith(".safetensors") and k.find("model") >= 0]
    if len(files) > 1:
        files = list(sorted(files, key=lambda x: int(x[6:11])))

    for file in files:
        f = os.path.join(src_folder, file)
        dst_f = os.path.join(dst_folder, file)
        src_dict = load_file(f)
        tgt_dict = {}
        for key, tensor in src_dict.items():
            if key.endswith(".qzeros"):
                print(f"Transforming tensor: {key}")
                pack_factor = 32 // bits
                qzeros = awq_to_marlin_zero_points(
                    tensor, tensor.shape[0], tensor.shape[1] * pack_factor, bits
                )
                tgt_dict[key] = qzeros
            else:
                tgt_dict[key] = tensor
        save_file(tgt_dict, dst_f)
    print("Transformation complete.")


import json


def load_json(json_path, fn):
    json_fn = os.path.join(json_path, fn)
    with open(json_fn, "r", encoding="utf-8") as f_json:
        json_dict = json.load(f_json)
    return json_dict


def main():
    """
    Main function to handle command-line arguments for transforming safetensors files.
    """
    parser = argparse.ArgumentParser(
        description="Transform AWQ zeros to Marlin format."
    )
    parser.add_argument(
        "--src",
        type=str,
        required=False,
        help="Path to the source safetensors single file.",
    )
    parser.add_argument(
        "--dst",
        type=str,
        required=True,
        help="Path to save the transformed safetensors file.",
    )

    parser.add_argument(
        "--bits", type=int, required=True, default=4, help="Weight bits."
    )

    args = parser.parse_args()
    assert args.src != "" and os.path.exists(args.src), (
        "Must provide src folder (or src folder not found)!"
    )
    assert args.dst != "" and not os.path.exists(args.dst), (
        "Must provide dst folder (or dst folder must be empty)!"
    )
    assert args.bits == 8 or args.bits == 4, (
        "only 4-bit and 8-bit models are supported!"
    )

    try:
        src_directory = args.src
        if not os.path.exists(args.dst):
            os.makedirs(args.dst)
        if os.path.exists(src_directory + "/model.safetensors.index.json"):
            shutil.copy2(src_directory + "/model.safetensors.index.json", args.dst)
        transform_file(args.src, args.dst, args.bits)
        shutil.copy2(src_directory + "/config.json", args.dst)
        shutil.copy2(src_directory + "/tokenizer.json", args.dst)
        shutil.copy2(src_directory + "/tokenizer_config.json", args.dst)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
