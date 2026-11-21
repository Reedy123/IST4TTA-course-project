"""独立从零复现入口：默认打印计划，显式执行时支持小测/完整四模式，不读取旧16组。"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from locked_source import verify, digest
from setup_runtime import DEFAULT_RUNTIME, RELEASE
from verify_inputs import check_inputs
from runtime_support import gpu_idle_info, wait_gpu_idle, monitor, process_environment, check_metrics

TOOLS = Path(__file__).resolve().parent


def make_plan(repo, output, python, modes, corruptions, seeds, all_180=False,
              smoke=False, max_batches=None, eval_images=None, device=0, master_port=16835):
    """延用锁定的四模式/调度/命令，仅扩展路径、设备和明确标注的小测范围。"""
    repo = Path(repo).expanduser().resolve()
    proof = verify(repo)
    output = Path(output).expanduser().resolve()
    protected = [repo, RELEASE]
    if (repo/'data').exists():protected.append((repo/'data').resolve())
    if output.exists() or any(output==p or p in output.parents or output in p.parents for p in protected):
        raise ValueError('输出必须位于源码/数据之外，且目录尚不存在')
    if device < 0 or not 1024 <= master_port <= 65535:
        raise ValueError('设备索引或master端口无效')
    batches = (2 if smoke else 0) if max_batches is None else max_batches
    limited = batches > 0
    images = (128 if limited else 10000) if eval_images is None else eval_images
    if batches < 0 or batches > 78 or not 1 <= images <= 10000:
        raise ValueError('小测适应批数为1..78，完整用0；评估样本数为1..10000')
    if smoke and not limited or not limited and images != 10000:
        raise ValueError('小测必须限制适应批数；完整运行保留10000张评估')
    if all_180 and limited:raise ValueError('小测不能与--all-180组合，避免误启动大量任务')
    python = str(Path(shutil.which(str(python)) or python).expanduser().absolute())
    sys.path.insert(0,str(repo))
    from experiments.complete_protocol import schedule, command, CORRUPTIONS
    from experiments.final_protocol import MODES
    if set(corruptions)-set(CORRUPTIONS) or set(modes)-set(MODES) or set(seeds)-{42,43,44}:
        raise ValueError('模式、扰动或适应随机种子不属于锁定配置')
    if all_180:modes,corruptions,seeds=list(MODES),CORRUPTIONS,[42,43,44]
    tasks=[]
    for task in schedule():
        if task['mode'] not in modes or task['corruption'] not in corruptions or task['seed'] not in seeds:continue
        adaptation=output/task['id']/'adaptation'
        cmd=command(repo,adaptation,task['mode'],task['corruption'],task['seed'],batches)
        cmd[0]=python;cmd[cmd.index('--master_port')+1]=str(master_port)
        if limited:
            pos=cmd.index(str(repo/'online_complete.py'))
            cmd[pos:pos+1]=[str(TOOLS/'adapt_smoke_entry.py'),'--locked-repo',str(repo),'--initial-eval-images',str(images)]
        ev=[python,'-m','experiments.complete_evaluate','--repo',str(repo),'--corruption',task['corruption'],
            '--training_seed',str(task['seed']),'--adapted','{checkpoint}','--out',str(output/task['id']/'evaluation')]
        if images<10000:ev=[python,str(TOOLS/'evaluate_subset.py'),*ev[3:],'--max-images',str(images)]
        tasks.append(dict(**task,adaptation_command=cmd,evaluation_command_template=ev))
    if not tasks:raise ValueError('筛选后没有任务')
    return dict(source=proof,kind='新发布小测' if limited else '新发布完整复现',
                smoke_only=limited,adaptation_batches=batches,evaluation_images=images,
                initial_evaluation_images=images,device=device,master_port=master_port,
                runtime=str(repo),repo=str(repo),output=str(output),python=python,
                legacy_records_required=False,evaluations='每个新模型重新评估，未复用旧准确率',
                validation='CPU检查不代表GPU验收，结果不能并入论文历史180/24组',
                smoke_difference='仅小测在内存中限制初始评估、适应批数及最终评估；完整运行不适配原入口',
                publication_git_commit=subprocess.check_output(['git','-C',str(RELEASE),'rev-parse','HEAD'],text=True).strip(),
                publication_tool_sha256={p.name:digest(p) for p in TOOLS.glob('*.py')},tasks=tasks)


def validate_environment(plan):
    """在所选解释器中检查版本及微型GPU联合算子，不安装或更改环境。"""
    command=[plan['python'],str(TOOLS/'doctor.py'),'--require-gpu','--device',str(plan['device'])]
    result=subprocess.run(command,capture_output=True,text=True)
    if result.returncode:raise ValueError('所选环境检查失败：'+result.stderr)
    return json.loads(result.stdout)


def evaluate_command(command, repo, log_path, device):
    """在指定GPU映射下执行独立评估，单独封装便于CPU模拟验证调度。"""
    with Path(log_path).open('x') as log:
        subprocess.run(command,cwd=repo,env=process_environment(device),stdout=log,stderr=subprocess.STDOUT,check=True)


def execute(plan):
    """只写全新结果目录；执行前检查输入/空闲GPU，失败保留日志且不自动续跑。"""
    repo,output=Path(plan['repo']),Path(plan['output'])
    verify(repo)
    if any(digest(TOOLS/name)!=sha for name,sha in plan['publication_tool_sha256'].items()):
        raise ValueError('发布工具在生成计划后发生变化，拒绝执行')
    if output.exists():raise ValueError('输出目录已存在，拒绝覆盖')
    if not Path(plan['python']).is_file():raise ValueError('所选Python解释器不存在')
    inputs=check_inputs(repo,sorted({t['corruption'] for t in plan['tasks']}))
    hardware=gpu_idle_info(plan['device'])
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        sock.bind(('127.0.0.1',plan['master_port']))
    output.mkdir(parents=True,exist_ok=False)
    (output/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
    (output/'inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2))
    results=[]
    try:
        environment=validate_environment(plan)
        (output/'environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2))
        for task in plan['tasks']:
            verify(repo);hardware=wait_gpu_idle(plan['device'],output/'gpu-safety.jsonl',check=gpu_idle_info)
            group=output/task['id'];group.mkdir()
            cmd=task['adaptation_command'];(group/'command.json').write_text(json.dumps(cmd,indent=2))
            timing=monitor(cmd,group/'adaptation.log',group/'gpu.csv',repo,plan['device'])
            (group/'launcher.json').write_text(json.dumps(timing,indent=2))
            if timing['exit_code']:raise ValueError('适应失败：'+task['id'])
            adaptation=group/'adaptation';weights=list(adaptation.glob('*adapt.pth'))
            if len(weights)!=1:raise ValueError('必须恰好有一个最终检查点')
            batches=[json.loads(line) for line in (adaptation/'profile-batches.jsonl').read_text().splitlines()]
            expected=plan['adaptation_batches'] or 79
            expected_images=plan['adaptation_batches']*128 if plan['smoke_only'] else 10000
            if len(batches)!=expected or sum(x['images'] for x in batches)!=expected_images:raise ValueError('适应批次/样本数不符')
            ev=[str(weights[0]) if x=='{checkpoint}' else x for x in task['evaluation_command_template']]
            (group/'evaluation-command.json').write_text(json.dumps(ev,indent=2))
            evaluate_command(ev,repo,group/'evaluation.log',plan['device'])
            metrics=json.loads((group/'evaluation/results.json').read_text());check_metrics(metrics,plan['evaluation_images'])
            profile=json.loads((adaptation/'profile-summary.json').read_text())
            elapsed=profile['adaptation_wall_seconds'];faiss=profile['stage_inclusive_seconds']['faiss']
            if not all(math.isfinite(x) and x>=0 for x in [elapsed,faiss]) or elapsed<=0:raise ValueError('计时数据无效')
            with (group/'evaluation/predictions.csv').open() as handle:
                if sum(1 for _ in csv.DictReader(handle))!=plan['evaluation_images']:raise ValueError('预测文件记录数不符')
            row=dict(id=task['id'],mode=task['mode'],corruption=task['corruption'],seed=task['seed'],
                     round=task['round'],attempt=str(group),checkpoint=str(weights[0]),checkpoint_sha256=digest(weights[0]),
                     checkpoint_seconds=profile['checkpoint_block_seconds'],peak_sampled_gpu_MiB=timing['peak_sampled_gpu_MiB'],
                     predictions=str(group/'evaluation/predictions.csv'),predictions_sha256=digest(group/'evaluation/predictions.csv'),
                     evaluation=metrics,single_percent=100*metrics['accuracy']['ist']['single'],eight_percent=100*metrics['accuracy']['ist']['eight'],
                     adaptation_seconds=elapsed,faiss_seconds=faiss,legacy_accuracy_reused=False,legacy_timing_reused=False,
                     publication_wrapper=True,original_suite_audit=False,smoke_only=plan['smoke_only'],
                     adaptation_images=expected_images,evaluation_images=plan['evaluation_images'],hardware=hardware)
            results.append(row)
            (output/'replication-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
        verify(repo)
        (output/'complete.json').write_text(json.dumps({'runs':len(results),'new_replication':True,'smoke_only':plan['smoke_only']},indent=2))
    except Exception as error:
        (output/'failure.json').write_text(json.dumps({'error':str(error),'complete':len(results)},ensure_ascii=False))
        raise


def main():
    """默认采用当前Python与旁置锁定工作树，兼容原--modes/--repo参数。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=DEFAULT_RUNTIME)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--modes','--mode',nargs='+',choices=['Original','Original-Reuse','No-history','Ours'],default=['Original','Original-Reuse','No-history','Ours'])
    parser.add_argument('--corruptions',nargs='+',default=['gaussian_noise'])
    parser.add_argument('--seeds',nargs='+',type=int,choices=[42,43,44],default=[42])
    parser.add_argument('--smoke',action='store_true',help='默认只适应2批，初始/最终各评估128张，结果仅用于流程检查')
    parser.add_argument('--max-batches',type=int)
    parser.add_argument('--eval-images',type=int)
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--master-port',type=int,default=16835)
    parser.add_argument('--all-180',action='store_true')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    try:
        plan=make_plan(args.repo,args.output,args.python,args.modes,args.corruptions,args.seeds,args.all_180,
                       args.smoke,args.max_batches,args.eval_images,args.device,args.master_port)
        print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
        if args.execute:execute(plan)
    except (ValueError,OSError,KeyError,subprocess.SubprocessError) as error:parser.exit(2,f'复现准备/执行失败：{error}\n')


if __name__=='__main__':
    main()
