"""在锁定源码之外执行24组交替计时，保持原命令及测量方式不变。"""
import json,hashlib,zipfile,subprocess,statistics,traceback,fcntl,tarfile,time
from pathlib import Path
from experiments.complete_protocol import command
from experiments.run_final_suite import monitor,source_inventory,write,attempt
from experiments.run_complete_suite import checkpoint_equal
from experiments.final_protocol import digest,records

home=Path.home();repo=home/'work/IST4TTA-course-project';old=home/'work/ist-complete-20261117-run02';root=home/'work/ist-efficiency24-20261118-run01'
corruptions=['gaussian_noise','motion_blur','fog','jpeg_compression']

def main():
    """先核验恢复文件和环境，再执行固定24组；成功组不重复运行。"""
    root.mkdir(exist_ok=True)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip()
    assert gpu=='NVIDIA GeForce RTX 3080',gpu
    protocol=json.loads((old/'protocol.json').read_text());assert source_inventory(repo)==protocol['source_files']
    incoming=home/'data/efficiency-incoming'
    expected={'ist-cifar-data.zip': 320474760, 'jpeg_compression.npy': 153600128, 'motion_blur.npy': 153600128, 'fog.npy': 153600128}
    while not all((incoming/n).exists() and (incoming/n).stat().st_size==size for n,size in expected.items()):
        write(root/'status.json',dict(complete=0,planned=24,status='waiting_for_data_upload'))
        time.sleep(30)
    (root/'runner-source.py').write_bytes(Path(__file__).read_bytes())
    with zipfile.ZipFile(incoming/'ist-cifar-data.zip') as z:
        assert z.testzip() is None
        for n in z.namelist():
            p=Path(n);assert not p.is_absolute() and '..' not in p.parts
            if n.endswith('/'):continue
            target=home/'data'/n;target.parent.mkdir(parents=True,exist_ok=True)
            b=z.read(n)
            if target.exists():assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(b).digest()
            else:target.write_bytes(b)
    for c in corruptions[1:]:
        src=incoming/(c+'.npy');assert digest(src)==protocol['data_sha256'][c]
        target=home/'data/CIFAR-10-C'/(c+'.npy')
        if target.exists():assert digest(target)==digest(src)
        else:target.hardlink_to(src)
    for c in corruptions:assert digest(repo/'data/CIFAR-10-C'/(c+'.npy'))==protocol['data_sha256'][c]
    for phase in ['env','weights','data']:
        with (root/('verify-'+phase+'.log')).open('w') as f:
            subprocess.run([str(home/'work/envs/ist310/bin/python'),'deploy/02_verify.py','--repo',str(repo),phase],cwd=repo,stdout=f,stderr=subprocess.STDOUT,check=True)
    tasks=[dict(corruption=c,repeat=k,mode=m) for k in range(3) for c in corruptions for m in (['Original','Ours'] if (k+corruptions.index(c))%2==0 else ['Ours','Original'])]
    write(root/'plan.json',dict(tasks=tasks,seed=42,source_files=protocol['source_files'],timing='与180组一致，仅最终保存；使用相同monitor和command',accuracy='每次逐张量匹配正式权重后复用其固定评估结果'))
    reference=json.loads((old/'results.json').read_text());rows=[]
    for t in tasks:
        ident=f"{t['corruption']}-r{t['repeat']+1}-{t['mode']}";group=root/ident
        if (group/'complete.json').exists():
            row=json.loads((group/'complete.json').read_text());assert digest(Path(row['checkpoint']))==row['checkpoint_sha256'];rows.append(row);continue
        out=attempt(group)
        try:
            assert source_inventory(repo)==protocol['source_files']
            write(root/'status.json',dict(complete=len(rows),planned=24,current=t,status='running'))
            cmd=command(repo,out/'adaptation',t['mode'],t['corruption'],42)
            write(out/'command.json',cmd)
            timing=monitor(cmd,out/'adaptation.log',out/'gpu.csv',repo);write(out/'launcher.json',timing);assert timing['exit_code']==0
            a=out/'adaptation';batches=records(a/'profile-batches.jsonl');assert len(batches)==79 and sum(b['images'] for b in batches)==10000
            checkpoint=next(a.glob('*adapt.pth'));ref=next(r for r in reference if r['corruption']==t['corruption'] and r['seed']==42 and r['mode']==t['mode'])
            checkpoint_equal(checkpoint,ref['checkpoint'])
            for name in ['input_trace.jsonl','faiss-audit.jsonl','pseudo-audit.jsonl']:
                assert (a/name).read_bytes()==(Path(ref['attempt'])/'adaptation'/name).read_bytes(),name
            p=json.loads((a/'profile-summary.json').read_text())
            row=dict(**t,checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),weights_exact=True,accuracy=ref['evaluation']['accuracy']['ist'],adaptation_seconds=p['adaptation_wall_seconds'],faiss_seconds=p['stage_inclusive_seconds']['faiss'],peak_MiB=timing['peak_sampled_gpu_MiB'])
            write(group/'complete.json',row);rows.append(row);write(root/'results.json',rows)
        except Exception:
            failure=dict(task=t,traceback=traceback.format_exc());write(out/'failure.json',failure);write(root/'status.json',dict(complete=len(rows),planned=24,status='failed',failure=failure));raise
    summary=[];lines=['# 24组效率稳定性补充实验','', '| 扰动 | 模式 | 适应秒均值±样本标准差 | FAISS秒均值±样本标准差 |','|---|---|---:|---:|']
    for c in corruptions:
        for m in ['Original','Ours']:
            rs=[r for r in rows if r['corruption']==c and r['mode']==m];d=dict(corruption=c,mode=m)
            for metric in ['adaptation_seconds','faiss_seconds']:
                values=[r[metric] for r in rs];d[metric]=dict(mean=statistics.mean(values),std=statistics.stdev(values))
            summary.append(d);lines.append(f"| {c} | {m} | {d['adaptation_seconds']['mean']:.2f} ± {d['adaptation_seconds']['std']:.2f} | {d['faiss_seconds']['mean']:.2f} ± {d['faiss_seconds']['std']:.2f} |")
        ratios=[]
        for k in range(3):
            a=next(r for r in rows if r['corruption']==c and r['repeat']==k and r['mode']=='Original');b=next(r for r in rows if r['corruption']==c and r['repeat']==k and r['mode']=='Ours');ratios.append(100*(1-b['adaptation_seconds']/a['adaptation_seconds']))
        lines.append(f"\n{c}：配对适应时间减少 {statistics.mean(ratios):.2f}% ± {statistics.stdev(ratios):.2f}%（ddof=1）。\n")
    write(root/'summary.json',summary);(root/'REPORT.md').write_text('\n'.join(lines)+'\n全部24组权重与对应正式结果逐张量一致；未重新评估或重跑180组，未修改锁定源码。\n')
    write(root/'status.json',dict(complete=24,planned=24,status='complete'))
    files=[dict(file=str(p.relative_to(root)),sha256=digest(p),bytes=p.stat().st_size) for p in root.rglob('*') if p.is_file() and p.suffix!='.zip' and p.name!='SHA256.json'];write(root/'SHA256.json',files)
    with zipfile.ZipFile(root/'backup.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in root.rglob('*'):
            if p.is_file() and p.suffix!='.zip':z.write(p,str(p.relative_to(root)))

if __name__=='__main__':
    with (home/'work/ist-efficiency24.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
