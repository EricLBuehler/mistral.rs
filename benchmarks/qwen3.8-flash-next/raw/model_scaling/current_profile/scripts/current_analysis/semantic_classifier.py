"""Semantic projection classification from source-verified kernel neighbors."""
import collections
import re

def is_projection(k):
    return (k.name.startswith('mmvq_gguf_') and '_quantize_' not in k.name) or ('void mul_mat_q<' in k.name)

def qformat(k):
    if 'mmvq_gguf_' in k.name:return k.name.split('mmvq_gguf_')[1].split('_bf16')[0]
    match=re.search(r'mul_mat_q<\(ggml_type\)(\d+)',k.name)
    return {'12':'q4_k','14':'q6_k','7':'q5_1','3':'q4_1'}.get(match[1]) if match else None

def quantify(kernels):
    semantic={};confidence={};hc_pairs=collections.Counter();bad_pairs=[]
    grouped=collections.defaultdict(list)
    for i,k in enumerate(kernels):grouped[(k.pid,k.context,k.stream)].append(i)
    def mark(i,label,why):
        if i in semantic and semantic[i]!=label:raise AssertionError((i,semantic[i],label))
        semantic[i]=label;confidence[i]=why
    def mark_projection(seq,p,label,why):
        mark(seq[p],label,why)
        if p+1<len(seq) and 'mul_mat_q_stream_k_fixup<' in kernels[seq[p+1]].name:mark(seq[p+1],label,why)
    for seq in grouped.values():
        mix_positions=[p for p,i in enumerate(seq) if 'q4_hc_mix_kernel' in kernels[i].name]
        for pos in mix_positions:
            # Only two adjacent projections with the known HC quant formats, with no intervening semantic boundary.
            previous=[]
            for q in range(pos-1,max(-1,pos-18),-1):
                k=kernels[seq[q]]
                if any(t in k.name for t in ['q4_hc_mix_kernel','q4_hc_combine_kernel','q4_hc_norm_kernel']):break
                if is_projection(k):
                    previous.append(q)
                    if len(previous)==2:break
            if len(previous)==2 and [qformat(kernels[seq[q]]) for q in previous]==['q5_1','q6_k']:
                for q,label in zip(previous,['hyper_up_projection','hyper_down_projection']):mark_projection(seq,q,label,'known formats and two projection groups immediately before hc_mix')
                hc_pairs['matched']+=1
            else:
                hc_pairs['unmatched']+=1
                if len(bad_pairs)<12:bad_pairs.append([kernels[seq[q]].name for q in previous])
            # Find the immediately following projection and its quantizer.
            nextp=None;quant=None
            for q in range(pos+1,min(len(seq),pos+15)):
                k=kernels[seq[q]]
                if 'quantize' in k.name:quant=q
                if is_projection(k):nextp=q;break
                if 'q4_hc_mix_kernel' in k.name or 'q4_hc_combine_kernel' in k.name:break
            if nextp is None:continue
            qk=kernels[seq[quant]] if quant is not None else None
            input10240=qk is not None and (('mmvq_gguf_quantize' in qk.name and qk.grid[0]==40) or ('quantize_mmq_q8_1<' in qk.name and qk.grid[1]==20))
            if input10240 and qformat(kernels[seq[nextp]])=='q6_k':
                mark_projection(seq,nextp,'hyper_inject_projection','first projection after hc_mix; activation quantizer confirms input width10240')
                stop=next((q for q in range(nextp+1,min(len(seq),pos+400)) if 'q4_hc_combine_kernel' in kernels[seq[q]].name or 'q4_hc_mix_kernel' in kernels[seq[q]].name),None)
                if stop is None or 'q4_hc_combine_kernel' not in kernels[seq[stop]].name:continue
                names=[kernels[seq[q]].name for q in range(nextp+1,stop)]
                if any('moe_router_topk' in n for n in names):
                    reduce_positions=[q for q in range(nextp+1,stop) if any(s in kernels[seq[q]].name for s in ['moe_weighted_reduce','moe_gemv_down_aggregate'])]
                    if reduce_positions:
                        last_reduce=reduce_positions[-1]
                        for q in range(last_reduce+1,stop):
                            if is_projection(kernels[seq[q]]):mark_projection(seq,q,'shared_expert_projection','within MLP hc_mix/combine, after expert reduction')
                elif any(any(s in n for s in ['gdn_','gated_delta_rule','causal_conv1d']) for n in names):
                    for q in range(nextp+1,stop):
                        if is_projection(kernels[seq[q]]):mark_projection(seq,q,'gdn_projection','within mixer hc_mix/combine with GDN kernels')
                elif any(any(s in n for s in ['q4_qsa_','qk_rms_norm_rope']) for n in names):
                    for q in range(nextp+1,stop):
                        if is_projection(kernels[seq[q]]):mark_projection(seq,q,'attention_projection','within mixer hc_mix/combine with QSA/RoPE kernels')
            elif quant is not None:
                input2560=(('mmvq_gguf_quantize' in qk.name and qk.grid[0]==10) or ('quantize_mmq_q8_1<' in qk.name and qk.grid[1]==5))
                if input2560 and qformat(kernels[seq[nextp]]) in ('q4_k','q6_k'):
                    mark_projection(seq,nextp,'vocabulary_head_projection','first projection after final hc_mix; input width2560 rather than inject10240')
        for pos,i in enumerate(seq):
            k=kernels[i]
            if i not in semantic and k.name.startswith('mmvq_gguf_') and '_plain_cuda' in k.name and k.grid[0]==124160:
                mark_projection(seq,pos,'vocabulary_head_projection','MMVQ output grid124160 is exactly248320 vocabulary rows /2')
    return semantic,confidence,dict(hc_pairs),bad_pairs

