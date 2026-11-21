"""小测通过后仅执行一次原版与资源复用的完整同期对照。"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
import torch


def main(root):
    """固定算法、数据和评估口径，每组独立启动并立即保存计时。"""
    root=Path(root);assert json.loads((root/'microbenchmark.json').read_text())['performance_pass']
    assert json.loads((root/'smoke-equivalence.json').read_text())['weight_tensors_exact']
    assert 'OK' in (root/'unit-tests.log').read_text()
    home=Path.home();repo=home/'work/IST4TTA-course-project';python=str(home/'work/envs/ist310/bin/python')
    source=repo/'data/ModelZoo/Hendrycks2020AugMixWRN.pt';source_hash=hashlib.sha256(source.read_bytes()).hexdigest();assert source_hash=='a39f020c48b3bad58df794b5d3a286724f00b9f026ee22d0e6d1710748599e82'
    spec=importlib.util.spec_from_file_location('monitor',home/'work/run_admission_ablation.py');monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0')
    config=dict(modes=['original','reuse-resources'],baseline='no-history',seed=42,batch=128,extend=8,K=50,actual_search_neighbors=52,memory=10000,lr=.001,iters=1,workers=2,images=10000,corruption='gaussian_noise',level=5,source_sha256=source_hash,index='GpuIndexFlatL2, default float32',reuse='StandardGpuResources only; new empty index each batch',timing='normal: synchronize at full adaptation boundaries; FAISS search returns original CPU results; include every-batch checkpoint saves',accuracy='existing fixed paired single/eight evaluator; official final random-view evaluation skipped in both',gate='7 unit tests, exact 8-batch check, replay>=1.2x and saved>=.02s/call already passed')
    (root/'locked-full-config.json').write_text(json.dumps(config,indent=2))
    files=subprocess.check_output(['git','ls-files','-co','--exclude-standard'],cwd=repo,text=True).splitlines()
    with tarfile.open(root/'full-source.tar.gz','w:gz') as z:
        for name in sorted(set(files)):
            p=repo/name
            if p.is_file() and p.stat().st_size<2_000_000 and not name.startswith('data/'):z.add(p,arcname=name)
    prior=home/'work/ist-ablation-a-20261115T132117Z';(root/'paired_eval.py').write_bytes((prior/'paired_eval.py').read_bytes());results={}
    for mode in config['modes']:
        group=root/'full'/mode;group.mkdir(parents=True,exist_ok=False);out=group/'adaptation'
        cmd=json.loads((root/'diagnostic-original/command.json').read_text());cmd.remove('--capture_faiss');cmd[cmd.index('--max_batches')+1]='0';cmd[cmd.index('--profile_kind')+1]='normal';cmd[cmd.index('--faiss_mode')+1]=mode;cmd[cmd.index('--save_path')+1]=str(out);cmd.append('--audit_faiss')
        (group/'command.json').write_text(json.dumps(cmd,indent=2));print('START full '+mode,flush=True)
        timing=monitor.monitor_run(cmd,group/'launcher.log',group/'gpu.csv',repo,env);(group/'launcher-timing.json').write_text(json.dumps(timing,indent=2));assert timing['exit_code']==0
        batches=monitor.read_records(out/'profile-batches.jsonl');faiss=monitor.read_records(out/'faiss-audit.jsonl');pseudo=monitor.read_records(out/'pseudo-audit.jsonl')
        assert len(batches)==len(faiss)==len(pseudo)==79 and sum(b['images'] for b in batches)==10000
        assert all(r['count']==r['index_count'] and r['neighbors']==52 for r in faiss)
        assert (out/'input_trace.jsonl').read_bytes()==(prior/'original/adaptation/input_trace.jsonl').read_bytes()
        checkpoint=next(out.glob('*adapt.pth'))
        evalcmd=[python,str(root/'paired_eval.py'),'--repo',str(repo),'--adapted',str(checkpoint),'--out',str(group/'evaluation')]
        (group/'evaluation-command.json').write_text(json.dumps(evalcmd,indent=2));eval_timing=monitor.monitor_run(evalcmd,group/'evaluation.log',group/'gpu-evaluation.csv',repo,env)
        (group/'evaluation-timing.json').write_text(json.dumps(eval_timing,indent=2));assert eval_timing['exit_code']==0
        evaluation=json.loads((group/'evaluation/results.json').read_text());assert evaluation['count']==10000 and evaluation['state_unchanged']
        profile=json.loads((out/'profile-summary.json').read_text())
        summary=dict(launcher=timing,profile=profile,evaluation=evaluation,evaluation_timing=eval_timing,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),complete_images=10000,adaptation_batches=79,training_updates=625)
        (group/'results.json').write_text(json.dumps(summary,indent=2));results[mode]=summary
        (root/'full-results-partial.json').write_text(json.dumps(results,indent=2));print('DONE full '+mode+' '+json.dumps(dict(adaptation=profile['adaptation_wall_seconds'],faiss=profile['stage_inclusive_seconds']['faiss'],accuracy=evaluation['accuracy']['ist'])),flush=True)
    a=root/'full/original/adaptation';b=root/'full/reuse-resources/adaptation'
    for file in ['input_trace.jsonl','faiss-audit.jsonl','pseudo-audit.jsonl','pseudo_quality.jsonl','admission.jsonl']:
        assert (a/file).read_bytes()==(b/file).read_bytes(),file+'不一致'
    first=torch.load(next(a.glob('*adapt.pth')),map_location='cpu');second=torch.load(next(b.glob('*adapt.pth')),map_location='cpu')
    history=torch.load(home/'work/ist-no-history-20261115T141107Z/adaptation/WRN-40-2_cifar10-c_gaussian_noise_5_adapt.pth',map_location='cpu')
    exact=all(torch.equal(first[k],second[k]) for k in first);historical=all(torch.equal(first[k],history[k]) for k in first);assert exact and historical
    assert (root/'full/original/evaluation/predictions.csv').read_bytes()==(root/'full/reuse-resources/evaluation/predictions.csv').read_bytes()
    first_results=results['original'];second_results=results['reuse-resources']
    old=first_results['profile']['adaptation_wall_seconds'];new=second_results['profile']['adaptation_wall_seconds'];oldf=first_results['profile']['stage_inclusive_seconds']['faiss'];newf=second_results['profile']['stage_inclusive_seconds']['faiss']
    summary=dict(groups=results,equivalence=dict(all_79_batches_exact=True,all_indices_distances_exact=True,all_pseudo_labels_exact=True,all_weight_tensors_exact=exact,historical_no_history_weights_exact=historical,fixed_evaluation_predictions_exact=True),comparison=dict(adaptation_saved_seconds=old-new,adaptation_reduction_fraction=(old-new)/old,adaptation_speedup=old/new,faiss_saved_seconds=oldf-newf,faiss_speedup=oldf/newf,warning='single contemporaneous pair; IO/loader jitter remains; no confidence interval or repeated full runs'))
    (root/'results.json').write_text(json.dumps(summary,indent=2));print('FULL_CONTROL_COMPLETE '+json.dumps(summary['comparison']),flush=True)

if __name__=='__main__':main(sys.argv[1])
