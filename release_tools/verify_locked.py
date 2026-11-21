"""命令行只读检查实验锁定版本，可在普通CPU电脑运行。"""
import argparse
import json
from locked_source import verify


def main():
    """输出核验结果，路径错误或内容改变时明确失败。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, help='独立检出的锁定实验工作树')
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.repo), ensure_ascii=False, indent=2))
    except (ValueError, OSError) as error:
        parser.exit(2, f'核验失败：{error}\n')


if __name__ == '__main__':
    main()
