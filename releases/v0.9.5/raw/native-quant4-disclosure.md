Native `--quant 4` is a separately quantized mixed-format configuration. It does not preserve the UD-Q4_K_XL weight bytes or precision distribution.

| Component | Native `--quant 4` | UD-Q4_K_XL GGUF |
|---|---|---|
| Token embedding and output head | Q6_K | Q8_0 |
| Dense attention/GDN projections | Q4_K | Mostly Q8_0; GDN a/b are F32 in the file |
| Shared experts | Q4_K gate/up; Q4_1 down | Q8_0 |
| Hyper-connection down/up/injection | Q6_K / Q5_1 / Q6_K | Q8_0 / Q8_0 / F32 |
| QSA indexer projection | Q6_K | F32 |
| PLE key/value projections | Q6_K | Q8_0 |
| Routed expert gate/up | Q4_K in all 48 layers | Q4_K in 47 layers, Q5_K in one |
| Routed expert down | Q4_1 in all 48 layers | Q5_1 in 43 layers, Q8_0 in five |
| PLE table | Custom symmetric Q4, groups of 32 with F16 scales | IQ4_NL |

The Q4_1 and Q5_1 choices follow existing shape fallbacks for widths that cannot use 256-element K blocks. Native and GGUF PLE both use 90 bytes per 160-element row, but dequantize to different values. The router/shared-expert gate remains dense. MRS loads the listed GGUF F32 projection tensors as BF16, while llama.cpp retains their F32 representation.

The cataloged text weights excluding PLE occupy 73.825 GB in native quantization versus 82.362 GB in MRS's GGUF representation. The nonexpert subset is 3.361 GB versus 5.344 GB. An idealized single-pass decode estimate, using 10 of 512 experts per layer, one token-embedding row and 16 PLE rows, is 4.216 GB versus 6.173 GB of active payload, about 31.7% less. These are logical payload estimates, not measured memory traffic or allocated VRAM; they exclude norms, vision, MTP, padding and repacks.

That lower active weight payload can materially improve decode speed. A native-versus-GGUF result cannot isolate engine optimization or establish equal model quality, even where quantization format names match. Keep the same-GGUF comparison separate. Confirm the native startup log selected the 4-bit PLE table before treating this inventory as the exact runtime configuration.

Final runtime confirmation: the native 8192 server used the mapped symmetric Q4 PLE table, logged as 26 GiB with 171 seconds of preparation. All 48 decoder layers were on CUDA. Startup completed in 540.4457 seconds.
