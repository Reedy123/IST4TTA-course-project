"""仅检查两批适应流程；临时入口不写回官方文件，输出明确标为 smoke。"""
import os
from pathlib import Path
import subprocess
import sys
import torch
import json

repo=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(exist_ok=False)
python=sys.executable
for mode in ['reference','original','confidence','consistency','combined']:
    source=(repo/('online.py' if mode=='reference' else 'online_filtered.py')).read_text()
    source=source.replace('for batch_idx, batch in enumerate(train_loader):','for batch_idx, batch in enumerate(train_loader):\n    if batch_idx >= 2: break')
    source=source.replace('# Test final model on the target dataset',"import sys\nsys.exit(0)\n# Test final model on the target dataset")
    script=out/(mode+'.py');script.write_text(source)
    args=[python,'-m','torch.distributed.launch','--nproc_per_node','1','--master_port','16349',str(script),
          '--dataset','cifar10-c','--dataset_path',str(repo/'data'),'--corruption','gaussian_noise','--level','5',
          '--model','WRN-40-2','--pretrained','--num_classes','10','--iters','1','--batch_size','32',
          '--num_workers','0','--max_len','1003','--save_model','--save_path',str(out/mode)]
    if mode!='reference':args+=['--admission',mode]
    env=dict(os.environ,PYTHONPATH=str(repo),PYTHONDONTWRITEBYTECODE='1')
    with (out/(mode+'.log')).open('w') as log:subprocess.run(args,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(mode,'SMOKE_PASS',flush=True)
a=torch.load(next((out/'reference').glob('*adapt.pth')),map_location='cpu')
b=torch.load(next((out/'original').glob('*adapt.pth')),map_location='cpu')
assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
records={}
for mode in ['original','confidence','consistency','combined']:
    records[mode]=[json.loads(s) for s in (out/mode/'admission.jsonl').read_text().splitlines()]
(out/'summary.json').write_text(json.dumps(dict(original_equals_reference=True,adaptation_batches=2,images=64,records=records),indent=2))
print('ORIGINAL_EXACT_STATE_MATCH',flush=True)
