from pathlib import Path
import json,hashlib,shutil,difflib
R=Path('/home/ericbuehler/mistral.rs');ROOT=Path(__file__).resolve().parent;OLD=Path('/home/ericbuehler/qwen4exp_work/real_routing_20260930/tile_variants');W=ROOT/'masked_y';B=W/'build';S=B/'shadow';S.mkdir(parents=True,exist_ok=False)
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
p=R/'mistralrs-quant/kernels/mmq_gguf/mmq_gguf.cuh';original=p.read_text();start=original.index('template <ggml_type type, int mmq_x, bool need_check, bool fixup>\nstatic __device__ __forceinline__ void mul_mat_q_process_tile(');end=original.index('// The mul_mat_q kernel implements',start);section=original[start:end]
old='const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop) {';assert section.count(old)==1
section=section.replace(old,'const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop, const bool mask_y_tail) {')
old='    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);';assert section.count(old)==1
section=section.replace(old,old+'\n    const int valid_y_ints = mask_y_tail ? min(mmq_x, tile_y_max_j + 1) * MMQ_TILE_Y_K : INT_MAX;')
assert section.count('tile_y[l] = by0[l];')==2;section=section.replace('tile_y[l] = by0[l];','tile_y[l] = l < valid_y_ints ? by0[l] : 0;')
new=original[:start]+section+original[end:]
old='tile_x_max_i, tile_y_max_j, 0, blocks_per_ne00.z);';assert new.count(old)==1;new=new.replace(old,'tile_x_max_i, tile_y_max_j, 0, blocks_per_ne00.z, ids_dst != nullptr);')
old='tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);';assert new.count(old)==2;new=new.replace(old,'tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop, ids_dst != nullptr);')
(B/p.name).write_text(new);(B/(p.name+'.base')).write_text(original)
patch=''.join(difflib.unified_diff(original.splitlines(True),new.splitlines(True),fromfile='a/'+str(p.relative_to(R)),tofile='b/'+str(p.relative_to(R))));(W/'masked_y.patch').write_text(patch)
d=json.loads((OLD/'build/provenance.json').read_text())
for key in ['build_started_at','build_finished_at','build_script_sha256','variant_binary','variant_binary_sha256','shadow_archive_sha256','compiled_objects','originals_unchanged','archive_command']:d.pop(key,None)
d['complete']=False;d['method']='External two-unit Q4K/Q4_1 shadow archive. Grouped MMQ masks unused activation columns to zero; dense path loads unchanged. Original tile shapes, weight loads, MMA accumulation, and full launch grid unchanged.'
d['original_sources'][str(p)]=sha(p);d['patched_sources']={str(B/p.name):sha(B/p.name)};d['patch_sha256']=sha(W/'masked_y.patch');d['grid_invariant']='Host tile selection, args.ncols_max, grid dimensions, and output assignments unchanged. Only grouped activation columns beyond each expert tile are zeroed instead of loaded.'
for command in d['nvcc_commands']:
 for i,arg in enumerate(command):
  if arg.startswith(str(OLD/'build')):command[i]=str(B)+arg[len(str(OLD/'build')):]
 source=next(Path(arg) for arg in command if arg.endswith('.cu'));shutil.copy2(R/'mistralrs-quant/kernels/mmq_gguf'/source.name,source);d['patched_sources'][str(source)]=sha(source)
d['rustc_command']=[str(S) if arg==str(OLD/'build/shadow') else arg for arg in d['rustc_command']]
shutil.copy2(OLD/'build/replay.source.rs',B/'replay.source.rs');shutil.copy2(OLD/'build_shadow.py',W/'build_shadow.py');(B/'provenance.json').write_text(json.dumps(d,indent=2)+'\n');print(W)
