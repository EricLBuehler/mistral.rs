# Remaining expert-kernel headroom

The current grouped kernel wastes substantial arithmetic on unused token columns. That does not establish an equally large time opportunity: after masked-Y, the measured L2 read-request volume is close to the logically selected compressed-weight payload. A selected-weight scan is needed before attributing the remaining time to avoidable computation.

## Exact route and arithmetic accounting

Each row is the median across five captures. M42 and M56 are native B6xQ7/B8xQ7 captures; M24 is the first three query positions from each of the eight sequences in the saved B8xQ7 captures. This is one captured layer, not a model-wide distribution.

| Rows | Routed assignments | Active experts | Token tile X | Computed token columns | Column MAC inflation | Full FFN MACs, useful / executed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 240 | 134 | 24 | 3,216 | 13.40x | 1.180G / 16.861G |
| 42 | 420 | 164 | 48 | 7,872 | 18.74x | 2.064G / 41.272G |
| 56 | 560 | 213 | 64 | 13,632 | 24.34x | 2.753G / 71.471G |

The launcher receives `ncols_max=num_tokens`, not the measured busiest expert. It chooses the first X producing one query tile. All captured expert counts are below X, so every active expert processes exactly one query tile. Y is 128 output features; both 640 and 2560 divide exactly. Gate/up K2560 is exact, while down K640 executes three 256-wide iterations, including a zero-activation 128-wide tail. This raises total MAC inflation to 14.29x/19.99x/25.97x. Counts exclude F32 scale/min corrections and decode work.

Every FFN launches 15,360 CTAs: 2,560 gate, 2,560 up, 10,240 down. Median nonempty CTA counts are 4,020/4,920/6,390. Empty CTAs skip matrix work. The previous compact scheduler already tested removing much of this empty-grid work and lost; the count alone is not a reason to repeat it.

Source anchors: `fast_mmq.rs:1461` passes the token count to both separate gate/up launches; `mmq_instance_q4_k.cu:10` defines Y/warps, `:63` selects the stream-K grid, and `:130` selects X. `mmq_gguf.cuh:3490` processes full tiles and `:1268` runs the NVIDIA integer-MMA path. `mmq_mma.cuh:876` emits `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`.

## What costs time and moves bytes

These are two different measurements and must remain labeled separately:

* Historical pre-mask recovered C8 Nsight Systems trace: 4,225 grouped FFNs. The three projections averaged 3,190.78 us; both activation quantizers together 17.12 us; dispatch count/prefix/scatter plus weighted reduction 28.69 us. The latter two categories are 1.44% of projection kernel time. Top-k routing averaged 5.66 us and also serves non-grouped FFNs. Allocation/API latency, gaps, and the dense router-logit projection are not included in those means. Nsight retained the warning that not all CUDA events might have been collected.
* Current masked isolated NCU capture, native M56 with 213 distinct experts: selected compressed weights are 610,713,600 bytes (582.42 MiB). Summed L2 read requests are 657,160,928 bytes, or 1.076x that selected payload. Gate/up are 1.061x each; down 1.103x. The requests also contain activations, metadata and alignment effects. They are not measured DRAM bytes, and no DRAM-bandwidth ceiling is inferred.

Per-expert compressed payload is 921,600 bytes for each Q4K gate/up and 1,024,000 bytes for Q4_1 down. There is no repeat of selected expert weights across query tiles at these native shapes. Gate/up input is quantized once per assignment, duplicating each token ten times, but this is already shared by both projections. Removing that duplication or fusing route metadata cannot by itself yield a large speedup in the historical profile.

The masked projection counters still show 254-255 registers/thread, 49,408 bytes shared per CTA, roughly14-16% tensor-pipe activity, and long-scoreboard stall ratios 2.17-3.30 per issued instruction. These counters support inspecting the projection implementation. They do not distinguish memory throughput, latency, decode, or instruction dependencies well enough to promise a gain. Previously increasing actual occupancy to two CTAs did not improve the pipeline.

As a sensitivity calculation only, doubling the grouped projections that occupied 56.25% of that historical summed kernel time would reduce that sum by 28.13%, a 1.39x ratio. It would not double serving throughput. The current final serving distribution is not assumed equal to the historical trace.

## One different design, not adopted

A warp-private integer-MMA projection is the next credible implementation only if the selected-weight scan demonstrates material time headroom. Keep the existing sorted routes, DS4 Q8 activation quantization and GGUF weights. Each warp computes 16 output features for up to 8 rows from one expert using the existing m16n8k32 integer instruction. Construct A fragments directly from a warp's packed Q4K/Q4_1 block bytes in registers; preserve the existing half-rounded weight scale/min products, Q8 scale/sum values, and ascending 32-element F32 correction order. Avoid the current decoded 128-feature X shared tile, its ldmatrix pass, and the full 64-token accumulator footprint.

Independent expert warps can share a small CTA, with a static rectangular mapping and an expert-local loop over actual 8-row groups. This does not require a tiny persistent grid, CPU counts, or putting different experts into one MMA tile. An MMA tile cannot directly combine independent expert operands without unwanted cross-expert products or block-diagonal padding. Pack independent work at warp/CTA level instead.

For the selected U213 M56 capture, 221 eight-row groups cover 560 assignments. This reduces token-column inflation from 24.34x to 3.16x. 207 of 213 experts, containing 482 assignments, fit a single 8-row group. If each group rereads its expert weights, logical weight replication is 221/213=1.038x. The median group replication across five native M56 samples is 1.038x, with the highly correlated sample reaching 1.212x. A useful implementation must preserve enough reuse for those hot experts.

This is a new integer-MMA data path, not another X selection, streamed-fragment change, compact/persistent scheduler, W4A16 cuTile decoder, or DP4A grouped GEMV. It reduces decode/staging and accumulator working sets while retaining tensor math. It is also considerably more work than the small accepted mask.

Risks that must be measured before adoption:

* A 16-feature warp that independently loads B repeats activation reads eight times as often as the current 128-feature CTA. On selected M56, ideal shared-load request payload would rise from 25.80 MB to 206.44 MB across the three projections without cache/shared reuse. Gate/up fusion or shared B staging could reduce this, but each adds complexity and may restore synchronization. Do not assume register-private loading is automatically faster.
* Direct fragment mapping and Q4K scale/min extraction must match the MMQ half rounding and correction ordering. Compare each projection and the complete FFN against preserved MMQ outputs and the independent FP32 oracle, then changed-route graphs and memcheck.
* Keeping entire packed 256-element weight blocks per warp can save reloads but raise register pressure. Compile resource checks must precede timing. A new kernel must improve paired whole-FFN time, not only occupancy or instruction count.
* Fusing gate/up alone saves input traffic and one launch, but not their independent weight payloads. Full gate/up/SiLU/down fusion needs a cross-feature reduction boundary and is not an assumed win.

Decision gate: finish the selected-weight scan first, paired on the same captures and cache conditions. If the scan is close to projection time, the practical opportunity is modest despite the MAC padding. If a substantial gap remains, prototype only the Q4K gate projection with direct integer fragments and require an improvement before adding Q4_1/down and full-FFN validation. No production change or new GPU measurement is included in this accounting.

## Matched cross-sequence reuse

[The paired route report](paired_sequence_reuse.md) compares each native batch with its own constituent sequences. Median selected-weight reuse is 1.384x for B6xQ7 and 1.469x for B8xQ7, reducing the logical selected payload by 27.75% and 31.95%, respectively. For B8 sample04, the separate sequence unions total 313 experts while the batch union has 213. These are matched route-set counts, not measured memory traffic or serving ceilings. Exact per-sequence expert IDs and bytes are in `paired_sequence_reuse.json`.

## Reproduction

Run `python3 build_accounting.py` at the original absolute paths. It reads only capture U32 routing IDs, metadata, saved profile JSON and source files. `accounting.json` contains every selected row index, all 512 expert occupancy counts, per-projection MAC/grid/byte counts, counter summaries and source SHA256 hashes. It does not read model weights or launch CUDA. `SHA256SUMS` covers this compact package.
