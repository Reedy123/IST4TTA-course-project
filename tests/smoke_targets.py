"""两种目标策略各两批GPU正确性检查；不用于正式准确率结论。"""
from pathlib import Path
import json
import os
import subprocess
import sys


def main():
    """检查目标形状、门控计数与相同输入随机状态。"""
    repo=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(exist_ok=False)
    reference=[json.loads(line) for line in (Path.home()/'work/ist-ablation-a-20261115T132117Z/original/adaptation/input_trace.jsonl').read_text().splitlines()][:2]
    summaries={}
    for mode in ['hard-only','selective-plca']:
        source=(repo/'online_targets.py').read_text()
        source=source.replace('for batch_idx, batch in enumerate(train_loader):','for batch_idx, batch in enumerate(train_loader):\n    if batch_idx >= 2: break')
        source=source.replace('# Test final model on the target dataset','import sys\nsys.exit(0)\n# Test final model on the target dataset')
        script=out/(mode+'.py');script.write_text(source)
        destination=out/mode
        command=[sys.executable,'-m','torch.distributed.launch','--nproc_per_node','1','--master_port','16353',str(script),
                 '--target_mode',mode,'--history_mode','no-history','--admission','original','--gate_confidence','0.8','--gate_agreement','0.75',
                 '--dataset','cifar10-c','--dataset_path','./data/','--corruption','gaussian_noise','--level','5','--model','WRN-40-2','--pretrained',
                 '--num_classes','10','--seed','42','--iters','1','--batch_size','128','--extend','8','--max_len','10000','--lr','0.001','--num_workers','2',
                 '--save_model','--save_path',str(destination)]
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(repo),CUDA_VISIBLE_DEVICES='0')
        with (out/(mode+'.log')).open('w') as f:subprocess.run(command,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        gate=[json.loads(line) for line in (destination/'target_gate.jsonl').read_text().splitlines()]
        trace=[json.loads(line) for line in (destination/'input_trace.jsonl').read_text().splitlines()]
        assert len(gate)==2 and sum(r['views'] for r in gate)==2048 and trace==reference
        if mode=='hard-only':assert all(not r['graph_executed'] and r['plca_proposed_changes']==0 for r in gate)
        else:assert all(r['accepted_changes']<=r['plca_proposed_changes'] and r['graph_executed'] for r in gate)
        summaries[mode]=dict(status='PASS',groups=256,views=2048,input_rng_exact=True,
                            proposed=sum(r['plca_proposed_changes'] for r in gate),accepted=sum(r['accepted_changes'] for r in gate))
        print(mode,json.dumps(summaries[mode]),flush=True)
    (out/'summary.json').write_text(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
