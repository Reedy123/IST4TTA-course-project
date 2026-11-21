"""为锁定算法生成从零复现计划；默认只打印命令，显式--execute才启动新实验。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from locked_source import verify, digest, MANIFESTS


def make_plan(repo, output, python, modes, corruptions, seeds, all_180=False):
    """复用锁定命令构造器和调度，不依赖旧16组或历史smoke证明。"""
    proof = verify(repo)
    output = Path(output).expanduser().resolve()
    repo = Path(repo).expanduser().resolve()
    if output.exists() or output == repo or repo in output.parents or output in repo.parents:
        raise ValueError('输出必须是锁定仓库之外、尚不存在的独立目录')
    sys.path.insert(0, str(repo))
    from experiments.complete_protocol import schedule, command, CORRUPTIONS
    if set(corruptions) - set(CORRUPTIONS):
        raise ValueError('扰动不在锁定的15种列表内')
    if all_180:
        modes = ['Original', 'Original-Reuse', 'No-history', 'Ours']
        corruptions, seeds = CORRUPTIONS, [42, 43, 44]
    tasks = []
    for task in schedule():
        if task['mode'] not in modes or task['corruption'] not in corruptions or task['seed'] not in seeds:
            continue
        adaptation = output / task['id'] / 'adaptation'
        cmd = command(repo, adaptation, task['mode'], task['corruption'], task['seed'])
        cmd[0] = str(Path(python).expanduser())
        evaluation = [cmd[0], '-m', 'experiments.complete_evaluate', '--repo', str(repo),
                      '--corruption', task['corruption'], '--training_seed', str(task['seed']),
                      '--adapted', '{checkpoint}', '--out', str(output / task['id'] / 'evaluation')]
        tasks.append(dict(**task, adaptation_command=cmd, evaluation_command_template=evaluation))
    if not tasks:
        raise ValueError('筛选后没有任务')
    return dict(source=proof, kind='发布后新增的从零复现包装，未用于历史实验',
                legacy_records_required=False, evaluations='每个新运行均重新评估，不复用旧16组准确率',
                limitations='未在GPU执行验收；无断点续跑；不替代原总控全部审计链或原180组历史记录',
                repo=str(repo), output=str(output), tasks=tasks)


def execute(plan):
    """仅在显式授权选项下执行新实验；失败保留文件并停止，不安装环境、不自动续跑。"""
    repo, output = Path(plan['repo']), Path(plan['output'])
    python = Path(plan['tasks'][0]['adaptation_command'][0])
    if not python.is_file():
        raise ValueError(f'实验解释器不存在：{python}')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
    if gpu != 'NVIDIA GeForce RTX 3080':
        raise ValueError('为保持论文计时硬件一致，仅允许单张RTX 3080；不自动改在3060上运行')
    protocol = json.loads((MANIFESTS / 'formal-protocol.json').read_text())
    if digest(repo / 'data/ModelZoo/Hendrycks2020AugMixWRN.pt') != protocol['source_weight_sha256']:
        raise ValueError('源权重摘要不匹配')
    for corruption in {t['corruption'] for t in plan['tasks']}:
        if digest(repo / 'data/CIFAR-10-C' / (corruption + '.npy')) != protocol['data_sha256'][corruption]:
            raise ValueError('扰动数据摘要不匹配：' + corruption)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    # 此处才启动Linux/GPU进程；命令生成和所有本地CPU检查均不会进入本函数。
    from experiments.run_final_suite import monitor
    results = []
    try:
        for task in plan['tasks']:
            verify(repo)
            group = output / task['id']
            group.mkdir()
            command = task['adaptation_command']
            (group / 'command.json').write_text(json.dumps(command, indent=2))
            timing = monitor(command, group / 'adaptation.log', group / 'gpu.csv', repo)
            if timing['exit_code'] != 0:
                raise ValueError('适应进程失败：' + task['id'])
            adaptation = group / 'adaptation'
            weights = list(adaptation.glob('*adapt.pth'))
            if len(weights) != 1:
                raise ValueError('每组必须恰好有一个最终检查点')
            cmd = [str(weights[0]) if x == '{checkpoint}' else x for x in task['evaluation_command_template']]
            (group / 'evaluation-command.json').write_text(json.dumps(cmd, indent=2))
            with (group / 'evaluation.log').open('x') as log:
                subprocess.run(cmd, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
            metrics = json.loads((group / 'evaluation/results.json').read_text())
            profile = json.loads((adaptation / 'profile-summary.json').read_text())
            # 新复现使用同一监控器采样，但不冒用历史结果或原总控的完整审计证明。
            results.append(dict(id=task['id'], mode=task['mode'], corruption=task['corruption'], seed=task['seed'],
                                round=task['round'], attempt=str(group),
                                checkpoint=str(weights[0]), checkpoint_sha256=digest(weights[0]),
                                checkpoint_seconds=profile['checkpoint_block_seconds'],
                                peak_sampled_gpu_MiB=timing['peak_sampled_gpu_MiB'],
                                predictions=str(group / 'evaluation/predictions.csv'),
                                predictions_sha256=digest(group / 'evaluation/predictions.csv'),
                                evaluation=metrics, single_percent=100 * metrics['accuracy']['ist']['single'],
                                eight_percent=100 * metrics['accuracy']['ist']['eight'],
                                adaptation_seconds=profile['adaptation_wall_seconds'],
                                faiss_seconds=profile['stage_inclusive_seconds']['faiss'],
                                legacy_accuracy_reused=False, legacy_timing_reused=False,
                                publication_wrapper=True, original_suite_audit=False))
            (output / 'replication-results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    except Exception as error:
        (output / 'failure.json').write_text(json.dumps({'error': str(error), 'complete': len(results)}, ensure_ascii=False))
        raise
    (output / 'complete.json').write_text(json.dumps({'runs': len(results), 'new_replication': True}))


def main():
    """以保守默认值生成四模式单场景计划，执行需要额外显式开关。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--python', default='~/work/envs/ist310/bin/python')
    parser.add_argument('--modes', nargs='+', choices=['Original', 'Original-Reuse', 'No-history', 'Ours'],
                        default=['Original', 'Original-Reuse', 'No-history', 'Ours'])
    parser.add_argument('--corruptions', nargs='+', default=['gaussian_noise'])
    parser.add_argument('--seeds', nargs='+', type=int, choices=[42, 43, 44], default=[42])
    parser.add_argument('--all-180', action='store_true', help='采用锁定调度的完整180组新复现计划')
    parser.add_argument('--execute', action='store_true', help='明确启动新GPU实验，默认不执行')
    args = parser.parse_args()
    try:
        plan = make_plan(args.repo, args.output, args.python, args.modes, args.corruptions, args.seeds, args.all_180)
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        if args.execute:
            execute(plan)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, f'复现准备/执行失败：{error}\n')


if __name__ == '__main__':
    main()
