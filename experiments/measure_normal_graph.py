"""用已有自然CPU回传边界测量正常PLCA总块，禁止额外阶段同步。"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import torch


def main(root):
    """只补一次同配置正常No-history测量，适应结束即退出。"""
    root=Path(root);home=Path.home();repo=home/'work/IST4TTA-course-project'
    assert (root/'checkpoint-equivalence.json').exists(),'先等待前四次剖析全部完成'
    group=root/'normal-graph/no-history';group.mkdir(parents=True,exist_ok=False);out=group/'adaptation'
    spec=importlib.util.spec_from_file_location('monitor',home/'work/run_admission_ablation.py');monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
    entry=root/'normal_graph_entry.py'
    text=(repo/'online_profile.py').read_text().replace("Path(__file__).parent / 'utils/pseudo_labeling.py'",repr(str(repo/'utils/pseudo_labeling.py'))).replace('profiler.stop()','profiler.stop()\nraise SystemExit(0)')
    entry.write_text(text)
    cmd=json.loads((root/'normal/no-history/command.json').read_text());cmd[cmd.index(str(repo/'online_profile.py'))]=str(entry);cmd[cmd.index('--save_path')+1]=str(out)
    (group/'command.json').write_text(json.dumps(cmd,indent=2))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(repo))
    timing=monitor.monitor_run(cmd,group/'launcher.log',group/'gpu.csv',repo,env);(group/'launcher-timing.json').write_text(json.dumps(timing,indent=2));assert timing['exit_code']==0
    summary=json.loads((out/'profile-summary.json').read_text());assert summary['stage_calls']['plca_total']==79
    batches=monitor.read_records(out/'profile-batches.jsonl');assert len(batches)==79 and sum(b['images'] for b in batches)==10000
    assert (out/'input_trace.jsonl').read_bytes()==(root/'normal/no-history/adaptation/input_trace.jsonl').read_bytes()
    first=torch.load(next((out).glob('*adapt.pth')),map_location='cpu');prior=torch.load(next((root/'normal/no-history/adaptation').glob('*adapt.pth')),map_location='cpu')
    assert all(torch.equal(first[k],prior[k]) for k in first),'正常PLCA总块观察改变了模型权重'
    (group/'equivalence.json').write_text(json.dumps(dict(inputs_rng_equal=True,all_weight_tensors_equal=True,additional_cuda_synchronization=False),indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main(sys.argv[1])
