# Native GGUF cuTile projection experiment

This prototype is not selected by the production MoE dispatcher. The original Q4K and Q4_1 QTensor device bytes remain the only persistent expert weight storage. Kernels load native block headers and packed nibbles, reconstruct each weight in FP32, round the complete weight once to BF16, and run BF16 tensor-core MMA with FP32 accumulation. Projection outputs are FP32 in original flat assignment order.

The primary oracle independently dequantizes the same original bytes on the CPU, rounds weights and inputs to BF16, and accumulates products in FP64 before producing FP32 output. A second reference uses unrounded dequantized FP32 weights to report the numerical effect of BF16 weight rounding separately. Existing grouped MMQ quantizes activations to Q8 and rounds some weight scale products to FP16; exact equivalence with that arithmetic is not claimed.

The projection tests cover every K position through one-hot inputs, all Q4K scale/min and nibble groups, Q4_1 K tails, output-column tails, multiple routing tile sizes, empty experts/high expert IDs, negative expert sentinel output, and repeated graph replay with changed activations and routes. The graph is warmed before capture; routing is rebuilt on the captured stream during each replay.

The host routing contract follows existing moe_align: the same BM must be used, padded count must fit allocated capacity and be BM-aligned, IDs are nonnegative assignment IDs or the assignments sentinel, every output assignment appears exactly once, experts are -1 or a valid local expert, and all buffers use the input CUDA device. The prototype checks tensor shapes, supported formats/tiles, and routing buffer lengths; it trusts the existing routing producer for device contents.

Full routed FFN replay will include alignment, both gate/up projections, FP32 SiLU and product, BF16 activation rounding, down projection, and the production FP32 weighted-reduction-to-BF16 operation. These complete isolated layer timings are not end-to-end model throughput.
