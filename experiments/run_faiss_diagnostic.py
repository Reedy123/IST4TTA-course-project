"""只运行原版FAISS的小规模分项诊断，证据保存后再决定是否优化。"""
from pathlib import Path
import datetime
import importlib.util
import json
import os
import subprocess
import tarfile


def main():
    """固定前8个真实批次，记录原始特征供后续同输入重放。"""
    home=Path.home();repo=home/'work/IST4TTA-course-project';root=home/'work'/('ist-faiss-reuse-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));root.mkdir()
    spec=importlib.util.spec_from_file_location('monitor',home/'work/run_admission_ablation.py');monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
    python=str(home/'work/envs/ist310/bin/python');group=root/'diagnostic-original';group.mkdir();out=group/'adaptation'
    for name,cmd in [('commit.txt',['git','rev-parse','HEAD']),('branch.txt',['git','branch','--show-current']),('status.txt',['git','status','--short']),('official-source.diff',['git','diff','--','online.py','utils','model','dataset','scripts']),('gpu.txt',['nvidia-smi']),('packages.txt',[python,'-m','pip','freeze'])]:
        (root/name).write_bytes(subprocess.check_output(cmd,cwd=repo))
    files=subprocess.check_output(['git','ls-files','-co','--exclude-standard'],cwd=repo,text=True).splitlines()
    with tarfile.open(root/'diagnostic-source.tar.gz','w:gz') as z:
        for name in sorted(set(files)):
            p=repo/name
            if p.is_file() and p.stat().st_size<2_000_000 and not name.startswith('data/'):z.add(p,arcname=name)
    cmd=[python,'-m','torch.distributed.launch','--nproc_per_node','1','--master_port','16576',str(repo/'online_faiss.py'),'--profile_kind','diagnostic','--profile_flow','no-history','--faiss_mode','original','--capture_faiss','--max_batches','8','--skip_final_eval','--history_mode','no-history','--admission','original','--dataset','cifar10-c','--dataset_path','./data/','--corruption','gaussian_noise','--level','5','--model','WRN-40-2','--pretrained','--num_classes','10','--seed','42','--iters','1','--batch_size','128','--extend','8','--max_len','10000','--lr','0.001','--num_workers','2','--save_model','--save_path',str(out)]
    (group/'command.json').write_text(json.dumps(cmd,indent=2));print('RUN_ROOT='+str(root),flush=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0')
    timing=monitor.monitor_run(cmd,group/'launcher.log',group/'gpu.csv',repo,env);(group/'launcher-timing.json').write_text(json.dumps(timing,indent=2));assert timing['exit_code']==0
    prior=home/'work/ist-ablation-a-20261115T132117Z/original/adaptation/input_trace.jsonl'
    assert (out/'input_trace.jsonl').read_bytes()==b''.join(prior.read_bytes().splitlines(keepends=True)[:8])
    summary=json.loads((out/'profile-summary.json').read_text());assert summary['stage_calls']['faiss_index_constructor_lazy_init']==8
    print('DIAGNOSTIC_COMPLETE '+json.dumps(summary),flush=True)

if __name__=='__main__':main()
