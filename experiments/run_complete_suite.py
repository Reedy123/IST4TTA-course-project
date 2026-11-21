"""执行锁定的180组计划；仅最终保存，按阶段续跑，完整记录失败原因。"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import traceback
import torch
from .complete_protocol import CORRUPTIONS, SEEDS, schedule, command
from .final_protocol import MODES, SOURCE_SHA, digest, records, exact_pair
from .run_final_suite import write, monitor, source_inventory, attempt
from .summarize_complete import summarize


def now():
    """统一使用带时区的UTC记录，报告展示时可换算北京时间。"""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def checkpoint_equal(a, b):
    """逐张量验证原始参数和BN状态；不依赖序列化文件字节巧合一致。"""
    x=torch.load(a,map_location='cpu');y=torch.load(b,map_location='cpu')
    assert x.keys()==y.keys() and all(torch.equal(x[k],y[k]) for k in x), '模型权重偏离旧结果'


def lock_protocol(root, repo, legacy):
    """锁定算法源码、全部数据、源权重、软件版本、任务顺序和评估协议。"""
    current=source_inventory(repo)
    data={n:digest(repo/'data/CIFAR-10-C'/(n+'.npy')) for n in CORRUPTIONS}
    assert digest(repo/'data/ModelZoo/Hendrycks2020AugMixWRN.pt')==SOURCE_SHA
    assert subprocess.check_output(['git','diff','--','online.py','utils','model','dataset','scripts'],cwd=repo)==b''
    packages=subprocess.check_output([str(Path.home()/'work/envs/ist310/bin/python'),'-m','pip','freeze'],text=True)
    if (root/'protocol.json').exists():
        cfg=json.loads((root/'protocol.json').read_text())
        assert cfg['source_files']==current and cfg['data_sha256']==data
        assert (root/'pip-freeze.txt').read_text()==packages
        assert cfg['legacy']==str(legacy)
        return cfg
    old=json.loads((legacy/'config.json').read_text())
    for name,value in old['source_files'].items():
        assert current.get(name)==value, '既有代码已变化：'+name
    oldrows=json.loads((legacy/'results.json').read_text())
    assert len(oldrows)==16 and json.loads((legacy/'all-complete.json').read_text())['groups']==16
    assert old['source_weights_sha256']==SOURCE_SHA and old['seed']==42
    for k,v in dict(batch=128,extend=8,K=50,memory=10000,lr=.001,iters=1,workers=2).items():
        assert old[k]==v
    oldmanifest=json.loads((legacy/'SHA256-FINAL.json').read_text())
    for row in oldmanifest['files']:
        assert digest(legacy/row['file'])==row['sha256'], '旧证据损坏：'+row['file']
    cfg=dict(created=now(),source_files=current,data_sha256=data,source_weight_sha256=SOURCE_SHA,
             legacy=str(legacy),corruptions=CORRUPTIONS,seeds=SEEDS,modes=MODES,plan=schedule(),
             seed_scope='42/43/44仅改变适应；最终固定评估增强种子始终42+原图索引',
             level=5,images=10000,batch=128,extend=8,K=50,actual_neighbors=52,memory=10000,
             lr=.001,iters=1,repeat=1,workers=2,checkpoint_policy='仅最后一批保存一份最终权重',
             timing='正常适应起止CUDA同步；含取数、日志、最终一次保存，排除初始测试及最终评估',
             legacy_reuse='16组仅复用准确率；全部180组按最终一次保存统一重测适应耗时；旧权重逐张量相同才免重新评估',
             memory_measurement='适应进程设备显存半秒采样峰值，非瞬时精确峰值',hourly_yuan=.93,
             estimated_hours=[9.1,11.4],estimated_compute_yuan=[8.46,10.60],storage_budget_GiB=4,
             statistics='每个扰动对3种子取均值和样本标准差ddof=1；总表先对15扰动取种子内均值，再对3种子统计；配对差按同扰动同种子计算')
    write(root/'protocol.json',cfg)
    (root/'pip-freeze.txt').write_text(packages)
    for name,cmd in [('git-commit.txt',['git','rev-parse','HEAD']),('git-branch.txt',['git','branch','--show-current']),
                     ('git-status.txt',['git','status','--short']),('working-tree.patch',['git','diff'])]:
        (root/name).write_bytes(subprocess.check_output(cmd,cwd=repo))
    paths=subprocess.check_output(['git','ls-files','-co','--exclude-standard'],cwd=repo,text=True).splitlines()
    with tarfile.open(root/'source-snapshot.tar.gz','x:gz') as tar:
        for n in sorted(set(paths)):
            p=repo/n
            if p.is_file() and p.stat().st_size<2000000 and not n.startswith('data/'):
                tar.add(p,arcname=n)
    (root/'gpu-initial.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
    write(root/'legacy-accuracy-inventory.json',oldrows)
    return cfg


def progress(root, cfg, rows, states):
    """每组完成或失败都原子保存状态；部分表明确标出未完成，不挑选结果。"""
    write(root/'results.json',rows)
    write(root/'status.json',dict(updated=now(),complete=len(rows),planned=180,states=states))
    summarize(root,rows,complete=len(rows)==180)


def audit_case(root, rows, corruption, seed):
    """某场景四模式齐全后，核查相同输入、RNG和两对复用等价性。"""
    by={r['mode']:r for r in rows if r['corruption']==corruption and r['seed']==seed}
    if len(by)!=4:return
    traces=[(Path(r['attempt'])/'adaptation/input_trace.jsonl').read_bytes() for r in by.values()]
    assert all(a==traces[0] for a in traces)
    assert len({r['evaluation']['input_sha256'] for r in by.values()})==1
    checks={}
    for a,b in [('Original','Original-Reuse'),('No-history','Ours')]:
        checks[a+'='+b]=exact_pair(Path(by[a]['attempt'])/'adaptation',Path(by[b]['attempt'])/'adaptation')
        assert Path(by[a]['predictions']).read_bytes()==Path(by[b]['predictions']).read_bytes()
    write(root/'audits'/f'{corruption}-s{seed}.json',checks)


def execute(root, repo, legacy):
    """串行执行固定顺序，不自动调参、删除失败尝试或因日志静默中止。"""
    root.mkdir(parents=True,exist_ok=True)
    cfg=lock_protocol(root,repo,legacy)
    (root/'audits').mkdir(exist_ok=True)
    previous=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
    states=previous.get('states',{t['id']:dict(status='pending') for t in cfg['plan']})
    rows=[]
    oldrows=json.loads((root/'legacy-accuracy-inventory.json').read_text())
    for task in cfg['plan']:
        ident=task['id'];group=root/'runs'/ident
        if (group/'complete.json').exists():
            row=json.loads((group/'complete.json').read_text())
            assert digest(Path(row['checkpoint']))==row['checkpoint_sha256']
            assert digest(Path(row['predictions']))==row['predictions_sha256']
            rows.append(row);states[ident]=dict(status='complete',reused_on_resume=True)
            audit_case(root,rows,task['corruption'],task['seed'])
            continue
        stage='preflight';out=None
        try:
            assert shutil.disk_usage(root).free>2*1024**3, '持久化空间不足2GiB，暂停而不删除数据'
            assert source_inventory(repo)==cfg['source_files'], '源码发生变化，拒绝继续'
            out=attempt(group);adapt=out/'adaptation';done=out/'adaptation-complete.json'
            states[ident]=dict(status='running',stage=stage,attempt=str(out),started=now())
            progress(root,cfg,rows,states)
            print('START '+ident+' '+str(out),flush=True)
            if not done.exists():
                stage='adaptation';states[ident]['stage']=stage
                write(root/'status.json',dict(updated=now(),complete=len(rows),planned=180,states=states))
                cmd=command(repo,adapt,task['mode'],task['corruption'],task['seed'])
                write(out/'adaptation-command.json',cmd)
                timing=monitor(cmd,out/'adaptation.log',out/'gpu-adaptation.csv',repo)
                write(out/'launcher-timing.json',timing)
                assert timing['exit_code']==0, '适应进程失败，详见adaptation.log'
                batches=records(adapt/'profile-batches.jsonl')
                assert len(batches)==79 and sum(b['images'] for b in batches)==10000
                faiss=records(adapt/'faiss-audit.jsonl')
                assert len(faiss)==79 and all(a['count']==a['index_count'] and a['neighbors']==52 for a in faiss)
                checkpoint=next(adapt.glob('*adapt.pth'))
                write(done,dict(checkpoint=str(checkpoint),sha256=digest(checkpoint),timing=timing))
            trained=json.loads(done.read_text());checkpoint=Path(trained['checkpoint'])
            assert digest(checkpoint)==trained['sha256']
            stage='evaluation';states[ident]['stage']=stage
            write(root/'status.json',dict(updated=now(),complete=len(rows),planned=180,states=states))
            old=next((r for r in oldrows if task['seed']==42 and r['corruption']==task['corruption'] and r['mode']==task['mode']),None)
            if old is not None:
                checkpoint_equal(checkpoint,old['checkpoint'])
                oldattempt=Path(old['attempt'])
                for name in ['input_trace.jsonl','faiss-audit.jsonl','pseudo-audit.jsonl']:
                    assert (adapt/name).read_bytes()==(oldattempt/'adaptation'/name).read_bytes(), '旧准确率复用检查失败：'+name
                source=next(p for p in oldattempt.glob('evaluation-*') if p.is_dir() and (p/'results.json').exists())
                target=out/'evaluation-reused'
                target.mkdir(exist_ok=True)
                for name in ['config.json','results.json','predictions.csv']:
                    if (target/name).exists():assert digest(target/name)==digest(source/name)
                    else:shutil.copy2(source/name,target/name)
                write(out/'evaluation-reuse.json',dict(source=str(source),legacy_checkpoint_sha256=old['checkpoint_sha256'],
                      current_checkpoint_sha256=trained['sha256'],all_weight_tensors_exact=True,legacy_timing_not_reused=True))
                evaluation=json.loads((target/'results.json').read_text())
                reused=True
            else:
                candidates=[p for p in sorted(out.glob('evaluation-*')) if p.is_dir()]
                valid=[p for p in candidates if (p/'results.json').exists()]
                if valid:target=valid[-1]
                else:
                    target=out/f'evaluation-{len(candidates)+1:03d}'
                    cmd=[str(Path.home()/'work/envs/ist310/bin/python'),'-m','experiments.complete_evaluate',
                         '--repo',str(repo),'--adapted',str(checkpoint),'--out',str(target),
                         '--corruption',task['corruption'],'--training_seed',str(task['seed'])]
                    write(out/(target.name+'-command.json'),cmd)
                    timing=monitor(cmd,out/(target.name+'.log'),out/(target.name+'-gpu.csv'),repo)
                    write(out/(target.name+'-timing.json'),timing)
                    assert timing['exit_code']==0, '评估失败，适应已保存，下次只续评估'
                evaluation=json.loads((target/'results.json').read_text());reused=False
            assert evaluation['count']==10000 and evaluation['state_unchanged']
            profile=json.loads((adapt/'profile-summary.json').read_text())
            row=dict(**task,attempt=str(out),checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
                     predictions=str(target/'predictions.csv'),predictions_sha256=digest(target/'predictions.csv'),
                     evaluation=evaluation,legacy_accuracy_reused=reused,legacy_timing_reused=False,
                     single_percent=100*evaluation['accuracy']['ist']['single'],eight_percent=100*evaluation['accuracy']['ist']['eight'],
                     adaptation_seconds=profile['adaptation_wall_seconds'],faiss_seconds=profile['stage_inclusive_seconds']['faiss'],
                     checkpoint_seconds=profile['checkpoint_block_seconds'],peak_sampled_gpu_MiB=trained['timing']['peak_sampled_gpu_MiB'])
            write(group/'complete.json',row);rows.append(row)
            states[ident]=dict(status='complete',attempt=str(out),finished=now())
            audit_case(root,rows,task['corruption'],task['seed'])
            progress(root,cfg,rows,states)
            print('DONE '+ident+' '+json.dumps({k:row[k] for k in ['single_percent','eight_percent','adaptation_seconds','faiss_seconds','legacy_accuracy_reused']}),flush=True)
        except Exception:
            failure=dict(status='failed',stage=stage,attempt=str(out),time=now(),traceback=traceback.format_exc())
            states[ident]=failure
            if out:write(out/'failure.json',failure)
            progress(root,cfg,rows,states)
            print('FAILED '+ident+' '+failure['traceback'],flush=True)
            raise
    assert len(rows)==180 and len(list((root/'audits').glob('*.json')))==45
    write(root/'all-complete.json',dict(groups=180,cases=45,legacy_accuracy_reused=sum(r['legacy_accuracy_reused'] for r in rows),
          legacy_timing_reused=False,finished=now()))
    progress(root,cfg,rows,states)
    print('ALL_COMPLETE '+str(root),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--legacy',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve();root.parent.mkdir(parents=True,exist_ok=True)
    proof=json.loads((Path(str(root)+'-smoke')/'complete.json').read_text())
    assert proof['source_files']==source_inventory(Path(__file__).resolve().parents[1])
    assert proof['seed42_four_modes_exact'] and proof['seeds43_44_launch_pass']
    with Path(str(root)+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execute(root,Path(__file__).resolve().parents[1],args.legacy.resolve())
