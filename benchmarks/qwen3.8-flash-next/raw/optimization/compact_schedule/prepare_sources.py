#!/usr/bin/env python3
"""Create an external compact-MMQ prototype from the finalized masked-Y header."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

REPO = Path('/home/ericbuehler/mistral.rs')
WORK = Path(__file__).resolve().parent
SOURCE = REPO / 'mistralrs-quant/kernels/mmq_gguf'
HEADER = 'mmq_gguf.cuh'
UNITS = {'q4_k': 'GGML_TYPE_Q4_K', 'q4_1': 'GGML_TYPE_Q4_1'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--masked-header-sha256', required=True)
    parser.add_argument('--output', type=Path, default=WORK / 'sources')
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve existing prototype source snapshots'
    assert sha(SOURCE / HEADER) == args.masked_header_sha256
    header = (SOURCE / HEADER).read_text()
    assert 'const bool mask_y_tail' in header
    assert header.count('tile_y[l] = l < valid_y_ints ? by0[l] : 0;') == 2
    snippet = (WORK / 'compact.snippet.cuh').read_text()
    marker = '#define DEFINE_MMQ_MOE_LAUNCHER'
    assert header.count(marker) == 1
    sources = {HEADER: header.replace(marker, snippet + '\n' + marker)}
    for suffix, kind in UNITS.items():
        filename = f'mmq_instance_{suffix}.cu'
        original = (SOURCE / filename).read_text()
        marker = f'  CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<{kind}, mmq_x, false>),'
        assert original.count(marker) == 1
        added = f'''  if (args.expert_bounds != nullptr && args.use_stream_k &&
      args.nchannels_y <= MMQ_COMPACT_MAX_EXPERTS &&
      args.ncols_y <= MMQ_COMPACT_MAX_ASSIGNMENTS &&
      nbytes_shared + MMQ_COMPACT_SHARED_INTS * sizeof(int) <= smpbo) {{
    // FIXUP_WORKSPACE reserves at least 128*128 floats; this prefix needs at most 1025 ints.
    int32_t *tile_prefix = reinterpret_cast<int32_t *>(tmp_fixup);
    mmq_compact_prefix<mmq_x><<<1, 1, 0, stream>>>(
        args.expert_bounds, tile_prefix, args.nchannels_y);
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q_compact<{kind}, mmq_x, false>),
                                 nbytes_shared);
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q_compact<{kind}, mmq_x, true>),
                                 nbytes_shared);
    const dim3 compact_grid(MMQ_COMPACT_BLOCKS_PER_SM * nsm);
    if (args.nrows_x % mmq_y == 0) {{
      mul_mat_q_compact<{kind}, mmq_x, false>
          <<<compact_grid, block_dims, nbytes_shared, stream>>>(
              args, tile_prefix, blocks_per_ne00_fd);
    }} else {{
      mul_mat_q_compact<{kind}, mmq_x, true>
          <<<compact_grid, block_dims, nbytes_shared, stream>>>(
              args, tile_prefix, blocks_per_ne00_fd);
    }}
    return;
  }}

'''
        sources[filename] = original.replace(marker, added + marker)
    args.output.mkdir(parents=True)
    original_hashes = {}
    patched_hashes = {}
    diff = []
    for filename, contents in sources.items():
        original = SOURCE / filename
        output = args.output / filename
        output.write_text(contents)
        original_hashes[str(original)] = sha(original)
        patched_hashes[str(output)] = sha(output)
        label = f'mistralrs-quant/kernels/mmq_gguf/{filename}'
        diff.extend(difflib.unified_diff(original.read_text().splitlines(True), contents.splitlines(True),
                                       fromfile='a/' + label, tofile='b/' + label))
    patch = args.output / 'compact_schedule.patch'
    patch.write_text(''.join(diff))
    metadata = dict(original_sources=original_hashes, patched_sources=patched_hashes,
                    patch_sha256=sha(patch), generator_sha256=sha(Path(__file__)),
                    snippet_sha256=sha(WORK / 'compact.snippet.cuh'),
                    scope='Grouped Q4K/Q4_1, <=640 packed assignment rows, <=1024 experts; unchanged host tile selection.',
                    prefix_capacity_bytes=4100, existing_min_fixup_capacity_bytes=65536,
                    arithmetic='Full-K, unchanged process_tile; same output-row/expert/tile ordering.',
                    production_source_modified=False, compiled=False)
    (args.output / 'sources.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(patch)


if __name__ == '__main__':
    main()
