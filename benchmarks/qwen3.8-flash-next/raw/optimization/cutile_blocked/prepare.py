from pathlib import Path
import difflib, hashlib, json
WORK=Path(__file__).resolve().parent
ROOT=Path('/home/ericbuehler/mistral.rs')
source=ROOT/'mistralrs-quant/src/cutile/gguf_moe.rs'
s=source.read_text()
assert hashlib.sha256(s.encode()).hexdigest()=='d9bba1f7fd899c99ed80b71ac86d18c251a6841b0faca115cb88ea3b7c677c40'
start=s.index('                    let block: Tile<i32, { [BK] }> =')
end=s.index('                    acc = mmaf(a, b, acc);',start)

def load(name, typ, shape, base, offset):
    dims=', '.join(shape); ones=', '.join('1' for _ in shape)
    return f'''let {name}_ptr: PointerTile<*mut {typ}, {{ [] }}> = pointer_to_tile({base});
let {name}_ptr: PointerTile<*mut {typ}, {{ [{ones}] }}> = {name}_ptr.reshape(const_shape![{ones}]);
let {name}_ptr: PointerTile<*mut {typ}, {{ [{dims}] }}> = {name}_ptr.broadcast(const_shape![{dims}]);
let {name}_ptr = {name}_ptr.offset_tile({offset});
let ({name}, _): (Tile<{typ}, {{ [{dims}] }}>, Token) = load_ptr_tko(
    {name}_ptr, ordering::Weak, None::<scope::TileBlock>, None, None, None, Latency::<0>,
);
'''
q='''let zero: Tile<i32, { [] }> = scalar_to_tile(0i32);
let one: Tile<i32, { [] }> = scalar_to_tile(1i32);
let b: Tile<bf16, { [BN, BK] }> = if FORMAT == 0 {
    let block: Tile<i64, { [BN] }> = exti(broadcast_scalar((kt * BK) / Q4K_VALUES, const_shape![BN]));
    let block_bytes: Tile<i64, { [BN] }> = exti(broadcast_scalar(Q4K_BYTES, const_shape![BN]));
    let byte_base: Tile<i64, { [BN] }> = (weight_base + block) * block_bytes;
    let half_base: Tile<i64, { [BN] }> = byte_base / broadcast_scalar(2i64, const_shape![BN]);
'''
q+=load('d','f16',['BN'],'weight_halves','half_base')
q+=load('m','f16',['BN'],'weight_halves','half_base + broadcast_scalar(1i64, const_shape![BN])')
q+='''let d: Tile<f32, { [BN] }> = convert_tile(d);
let m: Tile<f32, { [BN] }> = convert_tile(m);
let group: Tile<i32, { [4] }> = iota(const_shape![4]);
let group: Tile<i64, { [4] }> = exti(group);
let group: Tile<i64, { [BN, 4] }> = group.reshape(const_shape![1, 4]).broadcast(const_shape![BN, 4]);
let scale_base: Tile<i64, { [BN, 4] }> = byte_base.reshape(const_shape![BN, 1]).broadcast(const_shape![BN, 4]) + group;
'''
for name,off in [('s0',4),('s1',8),('s2',12)]:
 q+=load(name,'u8',['BN','4'],'weight_bytes',f'scale_base + broadcast_scalar({off}i64, const_shape![BN, 4])')
 q+=f'let {name}: Tile<i32, {{ [BN, 4] }}> = exti({name});\n'
q+='''let scale_lo: Tile<i32, { [BN, 4] }> = s0 & broadcast_scalar(SCALE_MASK, const_shape![BN, 4]);
let scale_hi: Tile<i32, { [BN, 4] }> = (s2 & broadcast_scalar(NIBBLE_MASK, const_shape![BN, 4]))
    | ((s0 >> broadcast_scalar(SCALE_BITS, const_shape![BN, 4])) << broadcast_scalar(NIBBLE_BITS, const_shape![BN, 4]));
let min_lo: Tile<i32, { [BN, 4] }> = s1 & broadcast_scalar(SCALE_MASK, const_shape![BN, 4]);
let min_hi: Tile<i32, { [BN, 4] }> = (s2 >> broadcast_scalar(NIBBLE_BITS, const_shape![BN, 4]))
    | ((s1 >> broadcast_scalar(SCALE_BITS, const_shape![BN, 4])) << broadcast_scalar(NIBBLE_BITS, const_shape![BN, 4]));
let scales: Tile<i32, { [BN, 8] }> = cat(scale_lo, scale_hi, 1i32);
let minima: Tile<i32, { [BN, 8] }> = cat(min_lo, min_hi, 1i32);
let scales: Tile<f32, { [BN, 8] }> = convert_tile(scales);
let minima: Tile<f32, { [BN, 8] }> = convert_tile(minima);
let scales: Tile<f32, { [BN, 8] }> = scales * d.reshape(const_shape![BN, 1]).broadcast(const_shape![BN, 8]);
let minima: Tile<f32, { [BN, 8] }> = minima * m.reshape(const_shape![BN, 1]).broadcast(const_shape![BN, 8]);
let scales: Tile<f32, { [BN, 4, 2] }> = scales.reshape(const_shape![BN, 4, 2]);
let minima: Tile<f32, { [BN, 4, 2] }> = minima.reshape(const_shape![BN, 4, 2]);
let scale_even: Tile<f32, { [BN, 4, 1] }> = extract(scales, [zero, zero, zero]);
let scale_odd: Tile<f32, { [BN, 4, 1] }> = extract(scales, [zero, zero, one]);
let min_even: Tile<f32, { [BN, 4, 1] }> = extract(minima, [zero, zero, zero]);
let min_odd: Tile<f32, { [BN, 4, 1] }> = extract(minima, [zero, zero, one]);
let quant_index: Tile<i32, { [128] }> = iota(const_shape![128]);
let quant_index: Tile<i64, { [128] }> = exti(quant_index);
let quant_offsets: Tile<i64, { [BN, 128] }> = byte_base.reshape(const_shape![BN, 1]).broadcast(const_shape![BN, 128])
    + quant_index.reshape(const_shape![1, 128]).broadcast(const_shape![BN, 128])
    + broadcast_scalar(16i64, const_shape![BN, 128]);
'''
q+=load('packed','u8',['BN','128'],'weight_bytes','quant_offsets')
q+='''let packed: Tile<u8, { [BN, 4, 32] }> = packed.reshape(const_shape![BN, 4, 32]);
let packed: Tile<i32, { [BN, 4, 32] }> = exti(packed);
let low: Tile<i32, { [BN, 4, 32] }> = packed & broadcast_scalar(NIBBLE_MASK, const_shape![BN, 4, 32]);
let low: Tile<f32, { [BN, 4, 32] }> = convert_tile(low);
let low: Tile<f32, { [BN, 4, 32] }> = low * scale_even.broadcast(const_shape![BN, 4, 32]) - min_even.broadcast(const_shape![BN, 4, 32]);
let low: Tile<bf16, { [BN, 4, 32] }> = convert_tile(low);
let high: Tile<i32, { [BN, 4, 32] }> = packed >> broadcast_scalar(NIBBLE_BITS, const_shape![BN, 4, 32]);
let high: Tile<f32, { [BN, 4, 32] }> = convert_tile(high);
let high: Tile<f32, { [BN, 4, 32] }> = high * scale_odd.broadcast(const_shape![BN, 4, 32]) - min_odd.broadcast(const_shape![BN, 4, 32]);
let high: Tile<bf16, { [BN, 4, 32] }> = convert_tile(high);
let low: Tile<bf16, { [BN, 4, 1, 32] }> = low.reshape(const_shape![BN, 4, 1, 32]);
let high: Tile<bf16, { [BN, 4, 1, 32] }> = high.reshape(const_shape![BN, 4, 1, 32]);
let decoded: Tile<bf16, { [BN, 4, 2, 32] }> = cat(low, high, 2i32);
let decoded: Tile<bf16, { [BN, 256] }> = decoded.reshape(const_shape![BN, 256]);
let section: Tile<i32, { [] }> = scalar_to_tile(kt % (Q4K_VALUES / BK));
let decoded: Tile<bf16, { [BN, BK] }> = extract(decoded, [zero, section]);
decoded
} else {
    let group: Tile<i32, { [GB] }> = iota(const_shape![GB]) + broadcast_scalar(kt * GB, const_shape![GB]);
    let group_count: Tile<i32, { [GB] }> = broadcast_scalar(k_size / Q4_1_VALUES, const_shape![GB]);
    let valid: Tile<bool, { [GB] }> = lt_tile(group, group_count);
    let zero_group: Tile<i32, { [GB] }> = broadcast_scalar(0i32, const_shape![GB]);
    let group: Tile<i32, { [GB] }> = select(valid, group, zero_group);
    let group: Tile<i64, { [GB] }> = exti(group);
    let blocks: Tile<i64, { [BN, GB] }> = weight_base.reshape(const_shape![BN, 1]).broadcast(const_shape![BN, GB])
        + group.reshape(const_shape![1, GB]).broadcast(const_shape![BN, GB]);
    let byte_base: Tile<i64, { [BN, GB] }> = blocks * broadcast_scalar(20i64, const_shape![BN, GB]);
    let half_base: Tile<i64, { [BN, GB] }> = byte_base / broadcast_scalar(2i64, const_shape![BN, GB]);
'''
q+=load('d','f16',['BN','GB'],'weight_halves','half_base')
q+=load('m','f16',['BN','GB'],'weight_halves','half_base + broadcast_scalar(1i64, const_shape![BN, GB])')
q+='''let d: Tile<f32, { [BN, GB] }> = convert_tile(d);
let m: Tile<f32, { [BN, GB] }> = convert_tile(m);
let d: Tile<f32, { [BN, GB, 1] }> = d.reshape(const_shape![BN, GB, 1]);
let m: Tile<f32, { [BN, GB, 1] }> = m.reshape(const_shape![BN, GB, 1]);
let quant_index: Tile<i32, { [16] }> = iota(const_shape![16]);
let quant_index: Tile<i64, { [16] }> = exti(quant_index);
let quant_offsets: Tile<i64, { [BN, GB, 16] }> = byte_base.reshape(const_shape![BN, GB, 1]).broadcast(const_shape![BN, GB, 16])
    + quant_index.reshape(const_shape![1, 1, 16]).broadcast(const_shape![BN, GB, 16])
    + broadcast_scalar(4i64, const_shape![BN, GB, 16]);
'''
q+=load('packed','u8',['BN','GB','16'],'weight_bytes','quant_offsets')
q+='''let packed: Tile<i32, { [BN, GB, 16] }> = exti(packed);
let low: Tile<i32, { [BN, GB, 16] }> = packed & broadcast_scalar(NIBBLE_MASK, const_shape![BN, GB, 16]);
let low: Tile<f32, { [BN, GB, 16] }> = convert_tile(low);
let low: Tile<f32, { [BN, GB, 16] }> = d.broadcast(const_shape![BN, GB, 16]) * low + m.broadcast(const_shape![BN, GB, 16]);
let low: Tile<bf16, { [BN, GB, 16] }> = convert_tile(low);
let high: Tile<i32, { [BN, GB, 16] }> = packed >> broadcast_scalar(NIBBLE_BITS, const_shape![BN, GB, 16]);
let high: Tile<f32, { [BN, GB, 16] }> = convert_tile(high);
let high: Tile<f32, { [BN, GB, 16] }> = d.broadcast(const_shape![BN, GB, 16]) * high + m.broadcast(const_shape![BN, GB, 16]);
let high: Tile<bf16, { [BN, GB, 16] }> = convert_tile(high);
let low: Tile<bf16, { [BN, GB, 1, 16] }> = low.reshape(const_shape![BN, GB, 1, 16]);
let high: Tile<bf16, { [BN, GB, 1, 16] }> = high.reshape(const_shape![BN, GB, 1, 16]);
let decoded: Tile<bf16, { [BN, GB, 2, 16] }> = cat(low, high, 2i32);
let decoded: Tile<bf16, { [BN, BK] }> = decoded.reshape(const_shape![BN, BK]);
decoded
};
let b_mask: Tile<bool, { [BN, BK] }> = valid_columns.reshape(const_shape![BN, 1]).broadcast(const_shape![BN, BK])
    & valid_k.reshape(const_shape![1, BK]).broadcast(const_shape![BN, BK]);
let b_zero: Tile<bf16, { [BN, BK] }> = constant(bf16::ZERO, const_shape![BN, BK]);
let b: Tile<bf16, { [BN, BK] }> = select(b_mask, b, b_zero);
let b: Tile<bf16, { [BK, BN] }> = permute(b, const_array![1, 0]);
'''
s=s[:start]+q+s[end:]
s=s.replace('        const BK: i32,','        const BK: i32,\n        const GB: i32,')
s=s.replace('        cfg.bk.to_string(),','        cfg.bk.to_string(),\n        (cfg.bk / 32).to_string(),')
for name in ['Q4K_QUANTS_OFFSET','QUANT_HEADERS_BYTES','Q4K_GROUP_VALUES','Q4K_GROUP_PAIR_VALUES','Q4K_LOW_SCALE_GROUPS','Q4_1_NIBBLE_VALUES']:
 import re
 s=re.sub(r'    const '+name+r': i32 = \d+;\n','',s)
s=s.replace('            let block_bytes = if FORMAT == 0 { Q4K_BYTES } else { Q4_1_BYTES };\n','')
s=s.replace('let byte_base: Tile<i64, { [BN, GB] }> = blocks * broadcast_scalar(20i64, const_shape![BN, GB]);', 'let bytes_per_block: Tile<i64, { [BN, GB] }> = exti(broadcast_scalar(Q4_1_BYTES, const_shape![BN, GB]));\nlet byte_base: Tile<i64, { [BN, GB] }> = blocks * bytes_per_block;')
p=WORK/'gguf_moe.rs';p.write_text(s)
import subprocess
subprocess.run(['rustfmt','--edition','2021',str(p)],check=True)
(WORK/'decoder.fragment.rs').write_text(q)
(WORK/'base.rs').write_text(source.read_text())
print(p)
