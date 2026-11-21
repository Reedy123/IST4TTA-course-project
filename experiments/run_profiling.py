"""在同一实例顺序测量两个既有流程，保存完整计时与复现证据。"""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import os
import subprocess
import tarfile
import torch


def main():
    """先检查单批次，再执行正常与诊断四次性能测量。"""
    home=Path.home();repo=home/'work/IST4TTA-course-project'
    root=home/'work'/('ist-plca-profile-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));root.mkdir()
    spec=importlib.util.spec_from_file_location('monitor',home/'work/run_admission_ablation.py');monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
    python=str(home/'work/envs/ist310/bin/python');env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0')
    prior=home/'work/ist-ablation-a-20261115T132117Z/original/adaptation/input_trace.jsonl'
    source=repo/'data/ModelZoo/Hendrycks2020AugMixWRN.pt'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='a39f020c48b3bad58df794b5d3a286724f00b9f026ee22d0e6d1710748599e82'
    for name,cmd in [('commit.txt',['git','rev-parse','HEAD']),('branch.txt',['git','branch','--show-current']),('status.txt',['git','status','--short']),('patch.diff',['git','diff']),('gpu.txt',['nvidia-smi']),('packages.txt',[python,'-m','pip','freeze'])]:
        (root/name).write_bytes(subprocess.check_output(cmd,cwd=repo))
    files=subprocess.check_output(['git','ls-files','-co','--exclude-standard'],cwd=repo,text=True).splitlines()
    with tarfile.open(root/'source-snapshot.tar.gz','w:gz') as z:
        for name in sorted(set(files)):
            p=repo/name
            if p.is_file() and p.stat().st_size<2_000_000 and not name.startswith('data/'):z.add(p,arcname=name)
    config=dict(seed=42,batch=128,extend=8,K=50,memory=10000,lr=.001,iters=1,workers=2,dataset='gaussian_noise',level=5,raw_images=10000,checkpoint_policy='save every adaptation batch, identical in all runs',normal_sync='adaptation start/end only',diagnostic_sync='full-device before/after GPU stages; CPU cg no added sync',order=['normal/no-history','normal/hard-only','diagnostic/hard-only','diagnostic/no-history'])
    (root/'config.json').write_text(json.dumps(config,indent=2));print('RUN_ROOT='+str(root),flush=True)
    def launch(kind,flow,smoke=False):
        """独立进程保证从相同权重及随机状态启动。"""
        group=root/('smoke' if smoke else kind)/flow;group.mkdir(parents=True);out=group/'adaptation'
        entry=repo/'online_profile.py'
        if smoke:
            entry=root/'smoke_entry.py'
            text=(repo/'online_profile.py').read_text().replace("Path(__file__).parent / 'utils/pseudo_labeling.py'",repr(str(repo/'utils/pseudo_labeling.py')))
            text=text.replace('    profiler.end_batch()','    profiler.end_batch()\n    break').replace('profiler.stop()','profiler.stop()\nraise SystemExit(0)')
            entry.write_text(text)
        cmd=[python,'-m','torch.distributed.launch','--nproc_per_node','1','--master_port','16475',str(entry),'--profile_kind',kind,'--profile_flow',flow,'--history_mode','no-history','--admission','original','--dataset','cifar10-c','--dataset_path','./data/','--corruption','gaussian_noise','--level','5','--model','WRN-40-2','--pretrained','--num_classes','10','--seed','42','--iters','1','--batch_size','128','--extend','8','--max_len','10000','--lr','0.001','--num_workers','2','--save_model','--save_path',str(out)]
        (group/'command.json').write_text(json.dumps(cmd,indent=2));print('START '+str(group.relative_to(root)),flush=True)
        timing=monitor.monitor_run(cmd,group/'launcher.log',group/'gpu.csv',repo,dict(env,PYTHONPATH=str(repo)))
        (group/'launcher-timing.json').write_text(json.dumps(timing,indent=2));assert timing['exit_code']==0,str(group)
        trace=(out/'input_trace.jsonl').read_bytes();expected=prior.read_bytes()
        if smoke:expected=expected.splitlines(keepends=True)[0]
        assert trace==expected,'输入或随机状态与原版偏离'
        batches=monitor.read_records(out/'profile-batches.jsonl');assert len(batches)==(1 if smoke else 79)
        assert sum(r['images'] for r in batches)==(128 if smoke else 10000)
        summary=json.loads((out/'profile-summary.json').read_text())
        if kind=='diagnostic':
            assert summary['stage_calls']['feature_forward']==(8 if smoke else 625)
            assert summary['stage_calls'].get('faiss',0)==((1 if smoke else 79) if flow=='no-history' else 0)
            assert summary['stage_calls'].get('scipy_cg',0)==((10 if smoke else 790) if flow=='no-history' else 0)
        print('DONE '+str(group.relative_to(root))+' '+json.dumps(summary),flush=True)
    for flow in ['no-history','hard-only']:launch('diagnostic',flow,True)
    for kind,flow in [('normal','no-history'),('normal','hard-only'),('diagnostic','hard-only'),('diagnostic','no-history')]:launch(kind,flow)
    equality={}
    for flow in ['no-history','hard-only']:
        normal=torch.load(next((root/'normal'/flow/'adaptation').glob('*adapt.pth')),map_location='cpu')
        diag=torch.load(next((root/'diagnostic'/flow/'adaptation').glob('*adapt.pth')),map_location='cpu')
        equality[flow]=all(torch.equal(normal[k],diag[k]) for k in normal)
        assert equality[flow],'计时改变了最终权重'
    (root/'checkpoint-equivalence.json').write_text(json.dumps(equality,indent=2))
    print('ALL_COMPLETE '+str(root),flush=True)

if __name__=='__main__':main()
