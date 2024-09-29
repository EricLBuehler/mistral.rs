import numpy as np

names = [
    # "PRE_cross_attention_mask",
    # "post_RI_cross_attention_mask",
    # "inverted_cross_attention_mask",
    # "BOOL_inverted_cross_attention_mask",
    # "filled_cross_attention_mask",
    "cross_attention_mask",
    "full_text_row_masked_out_mask",
]

for name in names:
    truth = np.load(f"truth/{name}.npy")
    mistralrs = np.load(f"mistralrs/{name}.npy")
    print("=" * 20, name, "=" * 20)
    print(f"{truth.shape=},{mistralrs.shape=}")
    print(f"{np.allclose(truth,mistralrs)}")
    print(f"{np.abs(truth-mistralrs).max()=}")
    print(f"{np.abs(truth-mistralrs).mean()=}")

