from pathlib import Path
import subprocess,re,json,hashlib
ROOT=Path(__file__).resolve().parent;B=Path('/home/ericbuehler/mistral.rs/target/release/build/mistralrs-quant-1c0ce6ee625e3844/out/kernels');W=ROOT/'masked_y_bounded';result=[]
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
for kind in ['q4_k','q4_1']:
 phases={};resources={}
 for phase,directory in [('baseline',B),('unbounded',ROOT/'masked_y/build/shadow'),('bounded',W/'build/shadow')]:
  obj=next(directory.glob(f'mmq_instance_{kind}-*.o'));text=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(obj)],text=True)
  functions={}
  for chunk in text.split('Function : ')[1:]:
   name,body=chunk.split('\n',1)
   if not name.startswith('_Z9mul_mat_qIL'):continue
   body=' '.join(body.split('........')[0].split());functions[name.strip()]=body
  phases[phase]=functions
  raw=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(obj)],text=True);(W/(kind+'.'+phase+'.resources.txt')).write_text(raw)
  resources[phase]={name:res.strip() for name,res in re.findall(r'Function ([^:]+):\n([^\n]+)',raw)}
 for name,body in phases['bounded'].items():
  m=re.match(r'_Z9mul_mat_qIL9ggml_type(\d+)ELi(\d+)ELb(\d)EE',name);assert m
  width=int(m[2]);expected='baseline' if width>64 else'unbounded'
  r=dict(quant=kind,tile=width,need_check=bool(int(m[3])),expected_equivalent=expected,sass_identical=body==phases[expected][name],sass_sha256={p:sha(d[name]) for p,d in phases.items()},resources={p:d[name] for p,d in resources.items()});result.append(r)
active=[r for r in result if r['tile'] < 48 or r['tile'] % 16 == 0]
assert all(r['sass_identical'] for r in active),[r for r in active if not r['sass_identical']]
(W/'code_equivalence.json').write_text(json.dumps(dict(method='Compare whitespace-normalized complete SASS bodies of active SM121f mul_mat_q templates: <=64 matches measured mask; >64 matches original baseline. Invalid width specializations contain only assertion stubs and are retained but excluded from equality requirement.',kernels=result),indent=2)+'\n');print('All',len(active),'active kernel SASS bodies match intended baseline or measured small-tile candidate.')
