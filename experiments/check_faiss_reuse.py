"""同特征重放与8批适应检查；不自动启动完整实验。"""
from pathlib import Path
import importlib.util
import json
import os
import sys
import time
import numpy as np
import torch
from .faiss_runtime import FaissRuntime


def benchmark(root):
    """两轮交替顺序，包含大小批次切换及每轮首次资源初始化。"""
    inputs=[]
    for path in sorted((root/'diagnostic-original/adaptation/faiss-inputs').glob('*.npz')):
        with np.load(path) as r:inputs.append((r['X'],int(r['neighbors'])))
    assert len(inputs)==8
    inputs.extend([(inputs[-1][0][:128].copy(),52),inputs[0]])
    records=[];expected={}
    for round_id,order in enumerate([['original','reuse-resources'],['reuse-resources','original']]):
        for mode in order:
            runtime=FaissRuntime(mode);torch.cuda.synchronize();start=time.perf_counter()
            for step,(X,k) in enumerate(inputs):
                resources=runtime.resources();index=runtime.index(resources,X.shape[1],__import__('faiss').GpuIndexFlatConfig(),'l2');runtime.add(index,X);result=runtime.search(index,X,k)
                key=(round_id,step)
                if key in expected:
                    assert np.array_equal(result[0],expected[key][0]) and np.array_equal(result[1],expected[key][1])
                else:expected[key]=(result[0].copy(),result[1].copy())
                del index;del resources
            del runtime;torch.cuda.synchronize();elapsed=time.perf_counter()-start
            records.append(dict(round=round_id,mode=mode,calls=len(inputs),seconds=elapsed))
    old=sum(r['seconds'] for r in records if r['mode']=='original');new=sum(r['seconds'] for r in records if r['mode']=='reuse-resources')
    decision=dict(records=records,speedup=old/new,saved_seconds_per_call=(old-new)/20,indices_distances_bitwise_equal=True,
                  locked_success_rule='FAISS replay speedup>=1.2 and saved>=0.02s/call; then require all 8-batch checks exact',performance_pass=old/new>=1.2 and (old-new)/20>=.02)
    (root/'microbenchmark.json').write_text(json.dumps(decision,indent=2));assert decision['performance_pass'],'小测收益不足，停止'


def main(root):
    """微基准通过后，两个已有流程独立适应前8批并严格对照。"""
    root=Path(root);benchmark(root);home=Path.home();repo=home/'work/IST4TTA-course-project'
    spec=importlib.util.spec_from_file_location('monitor',home/'work/run_admission_ablation.py');monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0');groups={}
    for mode in ['original','reuse-resources']:
        group=root/'smoke'/mode;group.mkdir(parents=True);out=group/'adaptation'
        cmd=json.loads((root/'diagnostic-original/command.json').read_text());cmd[cmd.index('--faiss_mode')+1]=mode;cmd[cmd.index('--save_path')+1]=str(out);cmd.append('--audit_faiss')
        (group/'command.json').write_text(json.dumps(cmd,indent=2));print('START smoke '+mode,flush=True)
        timing=monitor.monitor_run(cmd,group/'launcher.log',group/'gpu.csv',repo,env);(group/'launcher-timing.json').write_text(json.dumps(timing,indent=2));assert timing['exit_code']==0
        groups[mode]=out
    first=groups['original'];second=groups['reuse-resources']
    for filename in ['input_trace.jsonl','faiss-audit.jsonl','pseudo-audit.jsonl','pseudo_quality.jsonl']:
        assert (first/filename).read_bytes()==(second/filename).read_bytes(),filename+'不一致'
    arrays=0
    for folder in ['faiss-inputs','pseudo-inputs']:
        for path in sorted((first/folder).glob('*.npz')):
            with np.load(path) as a,np.load(second/folder/path.name) as b:
                for key in a.files:assert np.array_equal(a[key],b[key]),str(path)+key;arrays+=1
    a=torch.load(next(first.glob('*adapt.pth')),map_location='cpu');b=torch.load(next(second.glob('*adapt.pth')),map_location='cpu')
    assert all(torch.equal(a[k],b[k]) for k in a),'小测最终权重不一致'
    result=dict(images=1024,batches=8,training_updates=64,input_rng_exact=True,features_indices_distances_exact=True,pseudo_soft_targets_exact=True,weight_tensors_exact=True,compared_arrays=arrays)
    (root/'smoke-equivalence.json').write_text(json.dumps(result,indent=2));print('SMOKE_PASS '+json.dumps(result),flush=True)

if __name__=='__main__':main(sys.argv[1])
