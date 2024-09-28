import numpy as np

# names = [
#     "original_image",
#     "resize_image",
#     "pad_image",
#     "rescale_norm_image",
#     "aspect_ratio_ids",
#     "aspect_ratio_mask",
#     "split_image",
#     "packed_images",
#     "cross_attn_mask",
# ]

# for name in names:
#     truth = np.load(f"truth/{name}.npy")
#     mistralrs = np.load(f"mistralrs/{name}.npy")
#     print("=" * 20, name, "=" * 20)
#     print(f"{truth.shape=},{mistralrs.shape=}")
#     print(f"{np.allclose(truth,mistralrs)}")
#     print(f"{np.abs(truth-mistralrs).max()}")




# names = [
#     "vision_outputs",
#     #"attention_mask",
# ]

# for name in names:
#     truth = np.load(f"truth/{name}.npy")
#     mistralrs = np.load(f"mistralrs/{name}.npy")
#     print("=" * 20, name, "=" * 20)
#     print(f"{truth.shape=},{mistralrs.shape=}")
#     print(f"{np.allclose(truth,mistralrs)}")
#     print(f"{np.abs(truth-mistralrs).max()}")





# names = [
#     "patch_emb",
#     "tile_emb",
#     "cls_emb",
#     "ln_pre",
#     "pad_hs",
#     "transformer",
#     "ln_post",
#     "global_transformer",
#     "padrm",
#     "inter_hs_collected",
#     "inter_padrm",
#     "attention_mask",
# ]

# for name in names:
#     truth = np.load(f"truth/{name}.npy")
#     mistralrs = np.load(f"mistralrs/{name}.npy")
#     print("=" * 20, name, "=" * 20)
#     print(f"{truth.shape=},{mistralrs.shape=}")
#     print(f"{np.allclose(truth,mistralrs)}")
#     print(f"{np.abs(truth-mistralrs).max()=}")
#     print(f"{np.abs(truth-mistralrs).mean()=}")





# names = [
#     "patch_emb",
#     "tile_emb",
#     "cls_emb",
#     "ln_pre",
#     "pad_hs",
#     "attention_mask",
#     "layer1_beginattn",
#     "layer1_q_proj",
#     "layer1_k_proj",
#     "layer1_v_proj",
#     "layer1_attn_out",
#     "layer1_afterattn",
#     "layer1_afterres1",
#     "layer1_beginmlp",
#     "layer1_aftermlp",
# ]

# for name in names:
#     truth = np.load(f"truth/{name}.npy")
#     mistralrs = np.load(f"mistralrs/{name}.npy")
#     print("=" * 20, name, "=" * 20)
#     print(f"{truth.shape=},{mistralrs.shape=}")
#     print(f"{np.allclose(truth,mistralrs)}")
#     print(f"{np.abs(truth-mistralrs).max()=}")
#     print(f"{np.abs(truth-mistralrs).mean()=}")





names = [
    "pad_hs",
    "layer1_beginattn",
    # "layer1_q_proj",
    # "layer1_k_proj",
    # "layer1_v_proj",
    "layer1_qkT",
    "layer1_maskapply",
    "layer1_softmax",
    "layer1_attn_out",
    "layer1_afterattn",
    "layer1_afterres1",
    "layer1_beginmlp",
    "layer1_aftermlp",
]

for name in names:
    truth = np.load(f"truth/{name}.npy")
    mistralrs = np.load(f"mistralrs/{name}.npy")
    print("=" * 20, name, "=" * 20)
    print(f"{truth.shape=},{mistralrs.shape=}")
    print(f"{np.allclose(truth,mistralrs)}")
    print(f"{np.abs(truth-mistralrs).max()=}")
    print(f"{np.abs(truth-mistralrs).mean()=}")

