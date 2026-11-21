"""小测专用适配器：只在内存中限制初始评估样本，锁定训练源码文件保持不变。"""
import argparse
import ast
from pathlib import Path
import sys
from locked_source import verify


def smoke_tree(source, initial_images):
    """在唯一test_dataset赋值后限制初始评估；训练数据/损失/优化器均不改动。"""
    if not 1 <= initial_images <= 10000:
        raise ValueError('初始评估数量必须在1至10000之间')
    tree = ast.parse(source)
    positions = [i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'test_dataset' for t in node.targets)]
    if len(positions) != 1:
        raise ValueError('原入口结构不符合预期，拒绝改写或猜测')
    added = ast.parse(f'test_dataset = torch.utils.data.Subset(test_dataset, range(min({initial_images}, len(test_dataset))))').body[0]
    tree.body.insert(positions[0] + 1, added)
    return ast.fix_missing_locations(tree)


def main():
    """只用于max_batches>0的小测；完整运行直接使用原入口，无内存适配。"""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--locked-repo', type=Path, required=True)
    parser.add_argument('--initial-eval-images', type=int, required=True)
    args, remaining = parser.parse_known_args()
    if '--max_batches' not in remaining or int(remaining[remaining.index('--max_batches')+1]) < 1:
        parser.error('小测适配器必须明确限制max_batches，禁止用于正式完整运行')
    verify(args.locked_repo)
    source = args.locked_repo.resolve() / 'online_complete.py'
    tree = smoke_tree(source.read_text(), args.initial_eval_images)
    sys.path.insert(0, str(source.parent))
    sys.argv = [str(source), *remaining]
    exec(compile(tree, str(source), 'exec'), {'__name__': '__main__', '__file__': str(source)})


if __name__ == '__main__':
    main()
