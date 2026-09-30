from pathlib import Path
import difflib
import hashlib
import json
import subprocess

WORK = Path(__file__).resolve().parent
ROOT = Path('/home/ericbuehler/mistral.rs')
source = ROOT / 'mistralrs-quant/src/cutile/gguf_moe.rs'
s = source.read_text()
assert hashlib.sha256(s.encode()).hexdigest() == '9d92dca0b1e8aafc3071745198d375e9b3eccafe43fa0c6f2e1a86c793f454f4'
start = s.index('                    let zero: Tile<i32, { [] }> = scalar_to_tile(0i32);')
end = s.index('                    let b_mask:', start)


def load(name, typ, shape, base, offset):
    dims = ', '.join(shape)
    ones = ', '.join('1' for _ in shape)
    return f'''let {name}_ptr: PointerTile<*mut {typ}, {{ [] }}> = pointer_to_tile({base});
let {name}_ptr: PointerTile<*mut {typ}, {{ [{ones}] }}> = {name}_ptr.reshape(const_shape![{ones}]);
let {name}_ptr: PointerTile<*mut {typ}, {{ [{dims}] }}> = {name}_ptr.broadcast(const_shape![{dims}]);
let {name}_ptr = {name}_ptr.offset_tile({offset});
let ({name}, _): (Tile<{typ}, {{ [{dims}] }}>, Token) = load_ptr_tko(
    {name}_ptr, ordering::Weak, None::<scope::TileBlock>, None, None, None, Latency::<0>,
);
'''


q = '''let groups: Tile<i32, { [GB] }> = iota(const_shape![GB]) + broadcast_scalar(kt * GB, const_shape![GB]);
let group_count: Tile<i32, { [GB] }> = broadcast_scalar(k_size / Q4_1_VALUES, const_shape![GB]);
let valid_groups: Tile<bool, { [GB] }> = lt_tile(groups, group_count);
let groups: Tile<i32, { [GB] }> = select(valid_groups, groups, broadcast_scalar(0i32, const_shape![GB]));
let blocks: Tile<i32, { [GB] }> = if FORMAT == 0 {
    groups / broadcast_scalar(Q4K_SCALE_GROUPS, const_shape![GB])
} else {
    groups
};
let blocks: Tile<i64, { [GB] }> = exti(blocks);
let bytes_per_block = if FORMAT == 0 { Q4K_BYTES } else { Q4_1_BYTES };
let block_bytes: Tile<i64, { [BN, GB] }> = exti(broadcast_scalar(bytes_per_block, const_shape![BN, GB]));
let byte_base: Tile<i64, { [BN, GB] }> = (weight_base.reshape(const_shape![BN, 1]).broadcast(const_shape![BN, GB])
    + blocks.reshape(const_shape![1, GB]).broadcast(const_shape![BN, GB])) * block_bytes;
let half_base: Tile<i64, { [BN, GB] }> = byte_base / broadcast_scalar(2i64, const_shape![BN, GB]);
'''
q += load('d', 'f16', ['BN', 'GB'], 'weight_halves', 'half_base')
q += load('m', 'f16', ['BN', 'GB'], 'weight_halves', 'half_base + broadcast_scalar(1i64, const_shape![BN, GB])')
q += '''let d: Tile<f32, { [BN, GB] }> = convert_tile(d);
let m: Tile<f32, { [BN, GB] }> = convert_tile(m);
let local_group: Tile<i32, { [GB] }> = groups & broadcast_scalar(Q4K_SCALE_GROUPS - 1, const_shape![GB]);
let scale: Tile<f32, { [BN, GB] }> = if FORMAT == 0 {
    let lower: Tile<bool, { [GB] }> = lt_tile(local_group, broadcast_scalar(Q4K_LOW_SCALE_GROUPS, const_shape![GB]));
    let index: Tile<i32, { [GB] }> = local_group & broadcast_scalar(Q4K_LOW_SCALE_GROUPS - 1, const_shape![GB]);
    let index: Tile<i64, { [GB] }> = exti(index);
    let offsets: Tile<i64, { [BN, GB] }> = byte_base + broadcast_scalar(4i64, const_shape![BN, GB])
        + index.reshape(const_shape![1, GB]).broadcast(const_shape![BN, GB]);
'''
q += load('scale_low', 'u8', ['BN', 'GB'], 'weight_bytes', 'offsets')
q += load('scale_high', 'u8', ['BN', 'GB'], 'weight_bytes', 'offsets + broadcast_scalar(8i64, const_shape![BN, GB])')
q += '''let scale_low: Tile<i32, { [BN, GB] }> = exti(scale_low);
let scale_high: Tile<i32, { [BN, GB] }> = exti(scale_high);
let lower: Tile<bool, { [BN, GB] }> = lower.reshape(const_shape![1, GB]).broadcast(const_shape![BN, GB]);
let scale: Tile<i32, { [BN, GB] }> = select(lower,
    scale_low & broadcast_scalar(SCALE_MASK, const_shape![BN, GB]),
    (scale_high & broadcast_scalar(NIBBLE_MASK, const_shape![BN, GB]))
        | ((scale_low >> broadcast_scalar(SCALE_BITS, const_shape![BN, GB])) << broadcast_scalar(NIBBLE_BITS, const_shape![BN, GB])));
let scale: Tile<f32, { [BN, GB] }> = convert_tile(scale);
d * scale
} else {
    d
};
let bias: Tile<f32, { [BN, GB] }> = if FORMAT == 0 {
    let lower: Tile<bool, { [GB] }> = lt_tile(local_group, broadcast_scalar(Q4K_LOW_SCALE_GROUPS, const_shape![GB]));
    let index: Tile<i32, { [GB] }> = local_group & broadcast_scalar(Q4K_LOW_SCALE_GROUPS - 1, const_shape![GB]);
    let index: Tile<i64, { [GB] }> = exti(index);
    let offsets: Tile<i64, { [BN, GB] }> = byte_base + broadcast_scalar(8i64, const_shape![BN, GB])
        + index.reshape(const_shape![1, GB]).broadcast(const_shape![BN, GB]);
'''
q += load('min_low', 'u8', ['BN', 'GB'], 'weight_bytes', 'offsets')
q += load('min_high', 'u8', ['BN', 'GB'], 'weight_bytes', 'offsets + broadcast_scalar(4i64, const_shape![BN, GB])')
q += '''let min_low: Tile<i32, { [BN, GB] }> = exti(min_low);
let min_high: Tile<i32, { [BN, GB] }> = exti(min_high);
let lower: Tile<bool, { [BN, GB] }> = lower.reshape(const_shape![1, GB]).broadcast(const_shape![BN, GB]);
let minimum: Tile<i32, { [BN, GB] }> = select(lower,
    min_low & broadcast_scalar(SCALE_MASK, const_shape![BN, GB]),
    (min_high >> broadcast_scalar(NIBBLE_BITS, const_shape![BN, GB]))
        | ((min_low >> broadcast_scalar(SCALE_BITS, const_shape![BN, GB])) << broadcast_scalar(NIBBLE_BITS, const_shape![BN, GB])));
let minimum: Tile<f32, { [BN, GB] }> = convert_tile(minimum);
m * minimum
} else {
    m
};
let local: Tile<i32, { [BK] }> = safe_k & broadcast_scalar(block_values - 1, const_shape![BK]);
let quant_offset: Tile<i32, { [BK] }> = if FORMAT == 0 {
    broadcast_scalar(Q4K_QUANTS_OFFSET, const_shape![BK])
        + (local / broadcast_scalar(Q4K_GROUP_PAIR_VALUES, const_shape![BK])) * broadcast_scalar(Q4_1_VALUES, const_shape![BK])
        + (local & broadcast_scalar(Q4_1_VALUES - 1, const_shape![BK]))
} else {
    broadcast_scalar(QUANT_HEADERS_BYTES, const_shape![BK]) + (local & broadcast_scalar(NIBBLE_MASK, const_shape![BK]))
};
let quant_offset: Tile<i64, { [BK] }> = exti(quant_offset);
let expanded_base: Tile<i64, { [BN, BK] }> = byte_base.reshape(const_shape![BN, GB, 1])
    .broadcast(const_shape![BN, GB, 32]).reshape(const_shape![BN, BK]);
let quant_offset: Tile<i64, { [BN, BK] }> = expanded_base + quant_offset.reshape(const_shape![1, BK]).broadcast(const_shape![BN, BK]);
'''
q += load('packed', 'u8', ['BN', 'BK'], 'weight_bytes', 'quant_offset')
q += '''let packed: Tile<i32, { [BN, BK] }> = exti(packed);
let shift: Tile<i32, { [BK] }> = if FORMAT == 0 {
    ((local / broadcast_scalar(Q4_1_VALUES, const_shape![BK])) & broadcast_scalar(1i32, const_shape![BK]))
        * broadcast_scalar(NIBBLE_BITS, const_shape![BK])
} else {
    (local / broadcast_scalar(Q4_1_HALF_VALUES, const_shape![BK])) * broadcast_scalar(NIBBLE_BITS, const_shape![BK])
};
let shift: Tile<i32, { [BN, BK] }> = shift.reshape(const_shape![1, BK]).broadcast(const_shape![BN, BK]);
let q: Tile<i32, { [BN, BK] }> = (packed >> shift) & broadcast_scalar(NIBBLE_MASK, const_shape![BN, BK]);
let q: Tile<f32, { [BN, BK] }> = convert_tile(q);
let scale: Tile<f32, { [BN, BK] }> = scale.reshape(const_shape![BN, GB, 1])
    .broadcast(const_shape![BN, GB, 32]).reshape(const_shape![BN, BK]);
let bias: Tile<f32, { [BN, BK] }> = bias.reshape(const_shape![BN, GB, 1])
    .broadcast(const_shape![BN, GB, 32]).reshape(const_shape![BN, BK]);
let decoded: Tile<f32, { [BN, BK] }> = if FORMAT == 0 {
    scale * q - bias
} else {
    scale * q + bias
};
let b: Tile<bf16, { [BN, BK] }> = convert_tile(decoded);
'''
s = s[:start] + q + s[end:]
consts = '''    const Q4K_SCALE_GROUPS: i32 = 8;
    const Q4K_LOW_SCALE_GROUPS: i32 = 4;
    const Q4K_GROUP_PAIR_VALUES: i32 = 64;
    const Q4K_QUANTS_OFFSET: i32 = 16;
    const QUANT_HEADERS_BYTES: i32 = 4;
    const Q4_1_HALF_VALUES: i32 = 16;
'''
s = s.replace('    const NIBBLE_MASK:', consts + '    const NIBBLE_MASK:')
p = WORK / 'gguf_moe.rs'
p.write_text(s)
subprocess.run(['rustfmt', '--edition', '2021', str(p)], check=True)
(WORK / 'base.rs').write_text(source.read_text())
(WORK / 'grouped_metadata.patch').write_text(''.join(difflib.unified_diff(source.read_text().splitlines(True), p.read_text().splitlines(True), fromfile=str(source), tofile=str(source))))
(WORK / 'sources.json').write_text(json.dumps({'base_sha256':hashlib.sha256(source.read_bytes()).hexdigest(), 'candidate_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}, indent=2) + '\n')
print(p)
