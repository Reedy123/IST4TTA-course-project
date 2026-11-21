"""从本地Git锁定提交创建独立工作树并链接公开数据，不复制或改写算法。"""
import argparse
import json
from pathlib import Path
import subprocess
from locked_source import verify, MANIFESTS

RELEASE = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = RELEASE.parent / 'ist-experiment-locked'


def setup(runtime, data_dir):
    """拒绝覆盖已有非锁定目录或数据；不下载、不安装，不接触实验归档。"""
    runtime = Path(runtime).expanduser().resolve()
    data_dir = Path(data_dir).expanduser().resolve()
    if runtime == RELEASE or RELEASE in runtime.parents or runtime in RELEASE.parents:
        raise ValueError('运行目录必须独立于发布仓库')
    if data_dir == runtime or runtime in data_dir.parents or data_dir in runtime.parents:
        raise ValueError('数据目录必须位于锁定源码工作树之外')
    if not data_dir.is_dir():
        raise ValueError('请先准备数据目录（可以是空目录），setup不会代为下载：' + str(data_dir))
    if not runtime.exists():
        locked = json.loads((MANIFESTS / 'provenance.json').read_text())['locked_commit']
        subprocess.run(['git', '-c', 'core.autocrlf=false', '-C', str(RELEASE), 'worktree', 'add',
                        '--detach', str(runtime), locked], check=True)
    proof = verify(runtime)
    target = runtime / 'data'
    if target.is_symlink():
        if target.resolve() != data_dir:
            raise ValueError('已有data链接指向另一位置，不覆盖：' + str(target))
    elif target.exists():
        raise ValueError('已有data实体目录不覆盖；请选新的运行目录')
    else:
        target.symlink_to(data_dir, target_is_directory=True)
    return dict(source=proof, runtime=str(runtime), data_dir=str(data_dir), source_unchanged=True)


def main():
    """只创建独立工作树/数据链接；已有正确链接可重复检查。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument('--data-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(setup(args.runtime, args.data_dir), ensure_ascii=False, indent=2))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(2, f'准备失败：{error}\n')


if __name__ == '__main__':
    main()
