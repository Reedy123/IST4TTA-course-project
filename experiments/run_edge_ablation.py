"""固定β的四组完整实验；仅按事先规则触发两个控制组。"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
import torch


def paired(first,second):
    """只读比较统一评估器的逐样本预测，不参与算法决策。"""
    a=list(csv.DictReader(first.open()));b=list(csv.DictReader(second.open()));assert len(a)==len(b)==10000
    result={}
    for view in ['single','eight']:
        row=dict(disagreements=0,first_correct_second_wrong=0,first_wrong_second_correct=0)
        for x,y in zip(a,b):
            assert x['index']==y['index'] and x['label']==y['label'];p=x['ist_'+view];q=y['ist_'+view]
            row['disagreements']+=p!=q;row['first_correct_second_wrong']+=p==x['label'] and q!=x['label'];row['first_wrong_second_correct']+=p!=x['label'] and q==x['label']
        result[view]=row
    return result


def main(root):
    """从相同源权重逐组独立启动，所有文件写入新目录。"""
    root=Path(root);assert json.loads((root/'smoke-pass.json').read_text())['all_four_passed']
    home=Path.home();repo=home/'work/IST4TTA-course-project';python=str(home/'work/envs/ist310/bin/python');prior=home/'work/ist-faiss-reuse-20261117T013154Z'
    source=repo/'data/ModelZoo/Hendrycks2020AugMixWRN.pt';source_hash=hashlib.sha256(source.read_bytes()).hexdigest();assert source_hash=='a39f020c48b3bad58df794b5d3a286724f00b9f026ee22d0e6d1710748599e82'
    config=dict(beta=.5,formula='c=1-H(p)/ln(C); q=other-seven-view mean; a=1-JS(p,q)/ln(2); r=c,a,c*a; h=1-.5*(1-r); nonself A_ij*=h_j after original cube; preserve actual self indices; resymmetrize and renormalize unchanged',
                modes=['baseline','edge-c','edge-v','edge-cv'],conditional_controls=['uniform','shuffled-cv'],history='no-history',faiss='reuse-resources',seed=42,batch=128,extend=8,K=50,actual_neighbors=52,memory=10000,lr=.001,iters=1,workers=2,images=10000,corruption='gaussian_noise',level=5,source_sha256=source_hash,
                measurements='normal whole adaptation with original checkpoint/logging and equal new audits; CUDA sync only at adaptation boundaries; all modes compute reliability distributions; fixed paired evaluation',algorithm_uses_true_labels=False)
    (root/'locked-config.json').write_text(json.dumps(config,indent=2))
    for name,cmd in [('git-commit.txt',['git','rev-parse','HEAD']),('git-branch.txt',['git','branch','--show-current']),('git-status.txt',['git','status','--short']),('official-source.diff',['git','diff','--','online.py','utils','model','dataset','scripts']),('gpu.txt',['nvidia-smi']),('packages.txt',[python,'-m','pip','freeze'])]:
        (root/name).write_bytes(subprocess.check_output(cmd,cwd=repo))
    assert (root/'official-source.diff').stat().st_size==0
    paths=subprocess.check_output(['git','ls-files','-co','--exclude-standard'],cwd=repo,text=True).splitlines()
    with tarfile.open(root/'source-snapshot.tar.gz','x:gz') as z:
        for name in sorted(set(paths)):
            p=repo/name
            if p.is_file() and p.stat().st_size<2_000_000 and not name.startswith('data/'):z.add(p,arcname=name)
    (root/'paired_eval.py').write_bytes((prior/'paired_eval.py').read_bytes())
    spec=importlib.util.spec_from_file_location('monitor',home/'work/run_admission_ablation.py');monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0');results={};modes=list(config['modes'])
    for mode in modes:
        group=root/'full'/mode;group.mkdir(parents=True,exist_ok=False);out=group/'adaptation'
        cmd=json.loads((prior/'full/reuse-resources/command.json').read_text());cmd[cmd.index(str(repo/'online_faiss.py'))]=str(repo/'online_weighted_plca.py');cmd[cmd.index('--save_path')+1]=str(out);cmd+=['--edge_mode',mode]
        (group/'command.json').write_text(json.dumps(cmd,indent=2));print('START full '+mode,flush=True)
        timing=monitor.monitor_run(cmd,group/'launcher.log',group/'gpu.csv',repo,env);(group/'launcher-timing.json').write_text(json.dumps(timing,indent=2));assert timing['exit_code']==0,mode
        batches=monitor.read_records(out/'profile-batches.jsonl');edges=monitor.read_records(out/'edge-reliability.jsonl');cg=monitor.read_records(out/'cg-convergence.jsonl');faiss=monitor.read_records(out/'faiss-audit.jsonl');quality=monitor.read_records(out/'pseudo_quality.jsonl')
        assert len(batches)==len(edges)==len(faiss)==len(quality)==79 and sum(b['images'] for b in batches)==10000 and sum(r['views'] for r in edges)==80000
        assert len(cg)==790 and all(r['solution_finite'] and r['info']>=0 for r in cg)
        assert all(r['self_unchanged'] for r in edges) and all(r['count']==r['index_count'] and r['neighbors']==52 for r in faiss)
        assert all(r['history_entries_read']==0 for r in quality)
        assert (out/'input_trace.jsonl').read_bytes()==(prior/'full/reuse-resources/adaptation/input_trace.jsonl').read_bytes(),'输入或随机状态变化'
        checkpoint=next(out.glob('*adapt.pth'))
        if mode=='baseline':
            assert all(r['changed_nonself_edges']==0 for r in edges)
            previous=prior/'full/reuse-resources/adaptation'
            for name in ['faiss-audit.jsonl','pseudo-audit.jsonl']:
                assert (out/name).read_bytes()==(previous/name).read_bytes(),name
            a=torch.load(checkpoint,map_location='cpu');b=torch.load(next(previous.glob('*adapt.pth')),map_location='cpu');assert all(torch.equal(a[k],b[k]) for k in a),'完整基线回归失败'
        evalcmd=[python,str(root/'paired_eval.py'),'--repo',str(repo),'--adapted',str(checkpoint),'--out',str(group/'evaluation')]
        (group/'evaluation-command.json').write_text(json.dumps(evalcmd,indent=2));et=monitor.monitor_run(evalcmd,group/'evaluation.log',group/'gpu-evaluation.csv',repo,env);(group/'evaluation-timing.json').write_text(json.dumps(et,indent=2));assert et['exit_code']==0
        ev=json.loads((group/'evaluation/results.json').read_text());assert ev['count']==10000 and ev['state_unchanged'];assert ev['input_sha256']==json.loads((prior/'full/reuse-resources/evaluation/results.json').read_text())['input_sha256']
        profile=json.loads((out/'profile-summary.json').read_text());totals={key:sum(r[key] for r in quality) for key in ['groups','views','raw_group_correct','corrected_group_correct','raw_view_correct','corrected_view_correct','group_wrong_to_right','group_right_to_wrong','view_label_changes']}
        summary=dict(training=timing,profile=profile,evaluation=ev,evaluation_timing=et,quality=totals,cg=dict(calls=len(cg),converged=sum(r['info']==0 for r in cg),positive_info=sum(r['info']>0 for r in cg),maximum_relative_residual=max(r['relative_residual'] for r in cg)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),input_rng_exact=True)
        (group/'results.json').write_text(json.dumps(summary,indent=2));results[mode]=summary;(root/'results.json').write_text(json.dumps(results,indent=2))
        print('DONE full '+mode+' '+json.dumps(dict(accuracy=ev['accuracy']['ist'],adaptation=profile['adaptation_wall_seconds'],cg=summary['cg'])),flush=True)
        if mode=='edge-cv':
            count=lambda m,v:round(results[m]['evaluation']['accuracy']['ist'][v]*10000)
            conditions=dict(eight_minimum_20_extra_correct=count(mode,'eight')-count('baseline','eight')>=20,eight_above_c=count(mode,'eight')>count('edge-c','eight'),eight_above_v=count(mode,'eight')>count('edge-v','eight'),single_not_below_baseline=count(mode,'single')>=count('baseline','single'))
            proceed=all(conditions.values());(root/'control-decision.json').write_text(json.dumps(dict(conditions=conditions,run_controls=proceed,predeclared_rule=json.loads((root/'continuation-rule.json').read_text())),indent=2))
            print('CONTROL_DECISION '+json.dumps(dict(run_controls=proceed,conditions=conditions)),flush=True)
            if proceed:modes.extend(config['conditional_controls'])
    comparisons={}
    for first,second in [('baseline',m) for m in modes if m!='baseline']+[('edge-c','edge-cv'),('edge-v','edge-cv')]:
        comparisons[first+'_vs_'+second]=paired(root/'full'/first/'evaluation/predictions.csv',root/'full'/second/'evaluation/predictions.csv')
    (root/'paired-comparisons.json').write_text(json.dumps(comparisons,indent=2));(root/'complete.json').write_text(json.dumps(dict(modes=modes,images_per_mode=10000,all_complete=True),indent=2));print('ALL_COMPLETE '+str(root),flush=True)

if __name__=='__main__':main(sys.argv[1])
