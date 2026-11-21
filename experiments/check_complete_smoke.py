"""正式长任务前，验证仅最终保存不会改变种子42算法轨迹，并检查新种子入口。"""
import argparse
import json
from pathlib import Path
from .complete_protocol import command
from .final_protocol import MODES, exact_pair, records
from .run_final_suite import monitor, source_inventory, write


def main(root, legacy):
    """四模式各8批对齐旧小测，再验证种子43和44各一批。"""
    root.mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parents[1]
    inventory=source_inventory(repo)
    if (root/'complete.json').exists():
        assert json.loads((root/'complete.json').read_text())['source_files']==inventory
        return
    for seed,modes,batches in [(42,list(MODES),8),(43,['Ours'],1),(44,['Ours'],1)]:
        for mode in modes:
            group=root/f's{seed}-{mode}'
            if (group/'complete.json').exists():continue
            # 小测中断同样保留旧尝试，不覆盖已有日志和部分权重。
            group.mkdir(exist_ok=True)
            attempts=sorted(group.glob('attempt-*'))
            out=group/f'attempt-{len(attempts)+1:03d}';out.mkdir()
            adapt=out/'adaptation'
            cmd=command(repo,adapt,mode,'gaussian_noise',seed,batches)
            write(out/'command.json',cmd)
            print(f'SMOKE_START {seed} {mode}',flush=True)
            timing=monitor(cmd,out/'adaptation.log',out/'gpu.csv',repo)
            write(out/'timing.json',timing)
            assert timing['exit_code']==0
            assert len(records(adapt/'profile-batches.jsonl'))==batches
            assert (out/'adaptation.log').read_text().count('Save model to ')==1
            if seed==42:
                old=legacy/'gaussian_noise'/mode/'attempt-001/adaptation'
                exact_pair(adapt,old)
            write(group/'complete.json',dict(seed=seed,mode=mode,batches=batches,legacy_exact=seed==42))
            print(f'SMOKE_PASS {seed} {mode}',flush=True)
    write(root/'complete.json',dict(source_files=inventory,seed42_four_modes_exact=True,
          seeds43_44_launch_pass=True,checkpoint_saves_per_process=1))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--legacy-smoke',type=Path,required=True)
    args=parser.parse_args()
    main(args.root,args.legacy_smoke)
