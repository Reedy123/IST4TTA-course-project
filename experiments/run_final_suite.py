"""串行运行固定四扰动四模式，保存独立尝试并支持按已完成阶段续跑。"""
import argparse
import csv
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tarfile
import threading
import time
import torch
from .final_protocol import CORRUPTIONS, MODES, SOURCE_SHA, command, digest, exact_pair, records


def write(path, value):
    """原子更新状态文件，避免断线或进程退出留下半份JSON。"""
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def monitor(cmd, log, samples, repo):
    """仅监控退出状态和设备显存，不因静默中止正常计算。"""
    stop = threading.Event()
    def sample():
        """以半秒间隔记录设备显存和利用率，作为采样峰值。"""
        with samples.open('x', buffering=1) as f:
            f.write('timestamp,memory_MiB,utilization_percent\n')
            while not stop.is_set():
                p = subprocess.run(['nvidia-smi', '--query-gpu=timestamp,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], capture_output=True, text=True)
                if p.returncode == 0:
                    f.write(p.stdout)
                stop.wait(.5)
    thread = threading.Thread(target=sample)
    thread.start()
    start = time.monotonic()
    try:
        with log.open('x') as f:
            p = subprocess.run(cmd, cwd=repo, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0',
                               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1'), stdout=f, stderr=subprocess.STDOUT)
    finally:
        elapsed = time.monotonic()-start
        stop.set()
        thread.join()
    memory = [int(s.split(',')[1]) for s in samples.read_text().splitlines()[1:] if len(s.split(',')) == 3]
    return dict(exit_code=p.returncode, launcher_seconds=elapsed,
                peak_sampled_gpu_MiB=max(memory) if memory else None, sample_interval_seconds=.5)


def source_inventory(repo):
    """记录实际源码摘要，提交号以外的未提交改动也必须可追溯。"""
    names = subprocess.check_output(['git', 'ls-files', '-co', '--exclude-standard'], cwd=repo, text=True).splitlines()
    return {n: digest(repo/n) for n in sorted(set(names))
            if (repo/n).is_file() and n.endswith(('.py', '.sh')) and not n.startswith('data/')}


def initialize(root, repo, smoke):
    """固定配置、记录官方差异和环境；续跑时拒绝源码悄悄改变。"""
    inventory = source_inventory(repo)
    if (root/'config.json').exists():
        cfg = json.loads((root/'config.json').read_text())
        assert cfg['source_files'] == inventory and cfg['smoke'] == smoke
        return cfg
    assert not any(root.iterdir()), '已有未识别文件，拒绝覆盖'
    official = subprocess.check_output(['git', 'diff', '--', 'online.py', 'utils', 'model', 'dataset', 'scripts'], cwd=repo)
    assert official == b''
    cfg = dict(modes=MODES, corruptions=CORRUPTIONS if not smoke else ['gaussian_noise'], smoke=smoke,
               smoke_batches=8 if smoke else 0, seed=42, level=5, batch=128, extend=8, K=50,
               memory=10000, lr=.001, iters=1, workers=2, source_files=inventory,
               source_weights_sha256=SOURCE_SHA, adaptation_timing='起止CUDA同步，包含取数、日志和每批保存；排除初始和最终评估',
               faiss_timing='正常计时包围官方完整FAISS块，CPU结果返回；不叠加传输子时间',
               peak_memory='适应进程nvidia-smi半秒采样峰值，非精确瞬时峰值',
               resume='已完成组不重跑；完整适应成功但评估中断可只重做评估；不完整适应另建attempt',
               execution_order='按扰动轮换模式起始位置；每次独立进程，完整种子42')
    write(root/'config.json', cfg)
    for name, cmd in [('git-commit.txt', ['git','rev-parse','HEAD']),
                      ('git-branch.txt', ['git','branch','--show-current']),
                      ('git-status.txt', ['git','status','--short']),
                      ('working-tree.patch', ['git','diff'])]:
        (root/name).write_bytes(subprocess.check_output(cmd,cwd=repo))
    (root/'official-source.diff').write_bytes(official)
    (root/'pip-freeze.txt').write_bytes(subprocess.check_output([str(Path.home()/'work/envs/ist310/bin/python'),'-m','pip','freeze']))
    (root/'gpu.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
    paths = subprocess.check_output(['git','ls-files','-co','--exclude-standard'],cwd=repo,text=True).splitlines()
    with tarfile.open(root/'source-snapshot.tar.gz','x:gz') as tar:
        for n in sorted(set(paths)):
            p = repo/n
            if p.is_file() and p.stat().st_size < 2000000 and not n.startswith('data/'):
                tar.add(p, arcname=n)
    return cfg


def attempt(group):
    """只续用已完整适应的尝试，其余中断目录保留并另开新目录。"""
    group.mkdir(parents=True, exist_ok=True)
    existing = sorted(group.glob('attempt-*'))
    for p in reversed(existing):
        if (p/'adaptation-complete.json').exists():
            return p
    p = group/f'attempt-{len(existing)+1:03d}'
    p.mkdir(exist_ok=False)
    return p


def summarize(root, rows):
    """每完成一组立即更新汇总，避免依赖全部运行结束。"""
    write(root/'results.json', rows)
    fields = ['corruption','mode','single_percent','eight_percent','adaptation_seconds',
              'faiss_seconds','peak_sampled_gpu_MiB','checkpoint_sha256','attempt']
    with (root/'summary.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k:r[k] for k in fields})


def run(root, repo, smoke):
    """完成小测或唯一一套16组正式实验，不扩展扰动或调参。"""
    root.mkdir(parents=True, exist_ok=True)
    cfg = initialize(root, repo, smoke)
    assert digest(repo/'data/ModelZoo/Hendrycks2020AugMixWRN.pt') == SOURCE_SHA
    rows = []
    for ci, corruption in enumerate(cfg['corruptions']):
        modes = list(MODES)
        modes = modes[ci:]+modes[:ci]
        for mode in modes:
            group = root/corruption/mode
            if (group/'complete.json').exists():
                row = json.loads((group/'complete.json').read_text())
                assert digest(Path(row['checkpoint'])) == row['checkpoint_sha256']
                rows.append(row)
                summarize(root, rows)
                continue
            out = attempt(group)
            adapt = out/'adaptation'
            done = out/'adaptation-complete.json'
            if not done.exists():
                cmd = command(repo, adapt, mode, corruption, 8 if smoke else 0)
                write(out/'command.json', cmd)
                print(f'START {corruption} {mode} {out}', flush=True)
                timing = monitor(cmd, out/'adaptation.log', out/'gpu-adaptation.csv', repo)
                write(out/'launcher-timing.json', timing)
                assert timing['exit_code'] == 0, '适应失败，保留现场并停止'
                batches = records(adapt/'profile-batches.jsonl')
                assert len(batches) == (8 if smoke else 79)
                assert sum(b['images'] for b in batches) == (1024 if smoke else 10000)
                faiss = records(adapt/'faiss-audit.jsonl')
                assert len(faiss) == len(batches)
                assert all(r['count'] == r['index_count'] and r['neighbors'] == 52 for r in faiss)
                checkpoint = next(adapt.glob('*adapt.pth'))
                write(done, dict(checkpoint=str(checkpoint), sha256=digest(checkpoint), timing=timing))
            completed = json.loads(done.read_text())
            checkpoint = Path(completed['checkpoint'])
            assert digest(checkpoint) == completed['sha256']
            evaluation = None
            if not smoke:
                candidates = sorted(out.glob('evaluation-*'))
                valid = [p for p in candidates if (p/'results.json').exists()]
                if valid:
                    e = valid[-1]
                else:
                    e = out/f'evaluation-{len(candidates)+1:03d}'
                    cmd = [str(Path.home()/'work/envs/ist310/bin/python'), '-m', 'experiments.final_evaluate',
                           '--corruption', corruption, '--repo', str(repo), '--adapted', str(checkpoint), '--out', str(e)]
                    write(out/(e.name+'-command.json'), cmd)
                    timing = monitor(cmd, out/(e.name+'.log'), out/(e.name+'-gpu.csv'), repo)
                    write(out/(e.name+'-timing.json'), timing)
                    assert timing['exit_code'] == 0, '评估失败，保留已完成适应'
                evaluation = json.loads((e/'results.json').read_text())
                assert evaluation['count'] == 10000 and evaluation['state_unchanged']
            profile = json.loads((adapt/'profile-summary.json').read_text())
            row = dict(corruption=corruption, mode=mode, attempt=str(out), checkpoint=str(checkpoint),
                       checkpoint_sha256=digest(checkpoint), evaluation=evaluation,
                       single_percent=100*evaluation['accuracy']['ist']['single'] if evaluation else None,
                       eight_percent=100*evaluation['accuracy']['ist']['eight'] if evaluation else None,
                       adaptation_seconds=profile['adaptation_wall_seconds'],
                       faiss_seconds=profile['stage_inclusive_seconds']['faiss'],
                       peak_sampled_gpu_MiB=completed['timing']['peak_sampled_gpu_MiB'])
            write(group/'complete.json', row)
            rows.append(row)
            summarize(root, rows)
            print('DONE '+json.dumps(row, ensure_ascii=False), flush=True)
        by = {r['mode']:r for r in rows if r['corruption'] == corruption}
        audit = {}
        for a, b in [('Original','Original-Reuse'), ('No-history','Ours')]:
            audit[a+'='+b] = exact_pair(Path(by[a]['attempt'])/'adaptation', Path(by[b]['attempt'])/'adaptation')
            if not smoke:
                assert by[a]['evaluation']['input_sha256'] == by[b]['evaluation']['input_sha256']
                assert by[a]['evaluation']['accuracy'] == by[b]['evaluation']['accuracy']
        traces = [(Path(r['attempt'])/'adaptation/input_trace.jsonl').read_bytes() for r in by.values()]
        assert all(t == traces[0] for t in traces)
        if not smoke:
            assert len({r['evaluation']['input_sha256'] for r in by.values()}) == 1
            if corruption == 'gaussian_noise':
                historical = {
                    'Original': Path.home()/'work/ist-gaussian5-20261115T123720Z/results',
                    'Ours': Path.home()/'work/ist-no-history-20261115T141107Z/adaptation',
                }
                for mode, folder in historical.items():
                    old = torch.load(next(folder.glob('*adapt.pth')), map_location='cpu')
                    new = torch.load(by[mode]['checkpoint'], map_location='cpu')
                    assert old.keys() == new.keys() and all(torch.equal(old[k],new[k]) for k in old), mode+'偏离历史权重，停止'
                audit['historical_gaussian_weights_exact'] = True
        write(root/corruption/'equivalence.json', audit)
        print('EQUIVALENCE_PASS '+corruption, flush=True)
    write(root/'all-complete.json', dict(groups=len(rows), smoke=smoke, time=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print('ALL_COMPLETE '+str(root), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--validated-smoke', type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if not args.smoke:
        assert args.validated_smoke is not None, '正式实验必须先提供四模式小测结果'
        validation = json.loads((args.validated_smoke/'all-complete.json').read_text())
        assert validation['smoke'] and validation['groups'] == 4
        assert json.loads((args.validated_smoke/'config.json').read_text())['source_files'] == source_inventory(repo)
        assert (args.validated_smoke/'gaussian_noise/equivalence.json').exists()
    # 独立锁放在结果目录旁，防止两个tmux会话重复运行同一套实验。
    with Path(str(args.root)+'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.root.resolve(), repo, args.smoke)
