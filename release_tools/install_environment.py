"""按修复后Linux包导出安装新环境；默认仅打印步骤，不覆盖已有环境。"""
import argparse
import json
import platform
from pathlib import Path
import shutil
import subprocess

ENV_FILES = Path(__file__).resolve().parents[1] / 'reproduction/environment'


def installation_plan(prefix, conda='conda'):
    """使用显式conda包与pip来源锁，避免pip替换FAISS/MKL数值库。"""
    prefix = Path(prefix).expanduser().resolve()
    if prefix.exists():
        raise ValueError(f'环境路径已存在，拒绝覆盖：{prefix}')
    return [
        [conda, 'create', '-y', '-p', str(prefix), '--file', str(ENV_FILES / 'linux-64.explicit.txt')],
        [conda, 'run', '-p', str(prefix), 'python', '-m', 'pip', 'install', '--no-cache-dir',
         '-r', str(ENV_FILES / 'pip-runtime.txt')],
        [conda, 'run', '-p', str(prefix), 'python', '-m', 'pip', 'check'],
    ]


def main():
    """执行安装需显式--apply、Linux x86_64、可用conda以及全新日志目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--conda', default='conda')
    parser.add_argument('--record-dir', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    try:
        plan = installation_plan(args.prefix, args.conda)
        print(json.dumps({'platform': 'linux-64', 'commands': plan, 'apply_requested': args.apply}, ensure_ascii=False, indent=2), flush=True)
        if not args.apply:
            return
        if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
            raise ValueError('此包锁只适用Linux x86_64，不在Mac/ARM安装')
        if not shutil.which(args.conda):
            raise ValueError('找不到conda；请先使用实验室已有conda，不会自动安装')
        prefix = args.prefix.expanduser().resolve()
        record = (args.record_dir or prefix.with_name(prefix.name + '-install-record')).expanduser().resolve()
        if record == prefix or prefix in record.parents or record in prefix.parents:
            raise ValueError('日志目录必须独立于环境前缀')
        record.mkdir(parents=True, exist_ok=False)
        (record / 'plan.json').write_text(json.dumps(plan, indent=2))
        for i, cmd in enumerate(plan):
            with (record / f'step-{i+1}.log').open('x') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        for name, cmd in [('conda-explicit.txt', [args.conda, 'list', '-p', str(prefix), '--explicit']),
                          ('conda-list.json', [args.conda, 'list', '-p', str(prefix), '--json']),
                          ('pip-freeze.txt', [args.conda, 'run', '-p', str(prefix), 'python', '-m', 'pip', 'freeze'])]:
            with (record / name).open('x') as handle:
                subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=True)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(2, f'环境准备失败（保留已有日志，不自动删除环境）：{error}\n')


if __name__ == '__main__':
    main()
