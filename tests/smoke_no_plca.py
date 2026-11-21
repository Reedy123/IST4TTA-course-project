"""两批GPU正确性检查；不用于正式性能结论，不改写正式入口。"""
from pathlib import Path
import json
import os
import subprocess
import sys


def main():
    """临时截断适应循环，核对原始软目标和完整视图数量。"""
    repo=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(exist_ok=False)
    source=(repo/'online_no_plca.py').read_text()
    source=source.replace('for batch_idx, batch in enumerate(train_loader):','for batch_idx, batch in enumerate(train_loader):\n    if batch_idx >= 2: break')
    source=source.replace('# Test final model on the target dataset','import sys\nsys.exit(0)\n# Test final model on the target dataset')
    script=out/'smoke_entry.py';script.write_text(source)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(repo),CUDA_VISIBLE_DEVICES='0')
    command=[sys.executable,'-m','torch.distributed.launch','--nproc_per_node','1','--master_port','16352',str(script),
             '--history_mode','no-history','--plca_mode','no-plca','--admission','original',
             '--dataset','cifar10-c','--dataset_path','./data/','--corruption','gaussian_noise','--level','5','--model','WRN-40-2','--pretrained',
             '--num_classes','10','--seed','42','--iters','1','--batch_size','128','--extend','8','--max_len','10000','--lr','0.001','--num_workers','2',
             '--save_model','--save_path',str(out/'adaptation')]
    with (out/'launcher.log').open('w') as f:subprocess.run(command,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    rows=[json.loads(s) for s in (out/'adaptation/pseudo_quality.jsonl').read_text().splitlines()]
    assert len(rows)==2 and sum(r['groups'] for r in rows)==256
    assert all(r['views']==1024 and r['history_entries_read']==0 and r['view_label_changes']==0 and r['raw_view_correct']==r['corrected_view_correct'] for r in rows)
    result=dict(status='PASS',adaptation_batches=2,images=256,views=2048,raw_soft_targets_exact=True)
    (out/'smoke-result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    main()
