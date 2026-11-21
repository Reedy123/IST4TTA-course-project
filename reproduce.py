"""课程复现统一入口；默认运行计划不启动GPU，所有实验输出必须使用新目录。"""
import argparse
from pathlib import Path
import subprocess
import sys

TOOLS = {'setup': 'setup_runtime.py', 'install': 'install_environment.py', 'check': 'doctor.py',
         'weights': 'fetch_weights.py', 'inputs': 'verify_inputs.py', 'run': 'prepare_runs.py', 'stats': 'analyze_results.py',
         'summary': 'summarize_replication.py'}


def main():
    """按子命令转发参数，使用当前解释器，不依赖服务器固定Python路径。"""
    if len(sys.argv) > 1 and sys.argv[1] in TOOLS:
        script = Path(__file__).resolve().parent / 'release_tools' / TOOLS[sys.argv[1]]
        raise SystemExit(subprocess.call([sys.executable, str(script), *sys.argv[2:]]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=TOOLS, help='setup准备源码；install安装；check诊断；weights公开权重；inputs输入校验；run运行；stats历史重算；summary新结果汇总')
    parser.parse_args()


if __name__ == '__main__':
    main()
