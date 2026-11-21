"""用CPU只读核对源权重、所选扰动SHA-256及干净CIFAR文件存在性。"""
import argparse
import json
from pathlib import Path
from locked_source import verify, digest, MANIFESTS
from setup_runtime import DEFAULT_RUNTIME


def check_inputs(repo, corruptions=None):
    """保留原只读检查逻辑，并供新运行入口复用；不改写任何输入。"""
    repo = Path(repo).expanduser().resolve()
    proof = verify(repo)
    protocol = json.loads((MANIFESTS / 'formal-protocol.json').read_text())
    corruptions = corruptions or protocol['corruptions']
    if set(corruptions) - set(protocol['corruptions']):
        raise ValueError('扰动名称不属于锁定协议')
    expected = {'data/ModelZoo/Hendrycks2020AugMixWRN.pt': protocol['source_weight_sha256']}
    expected.update({'data/CIFAR-10-C/' + c + '.npy': protocol['data_sha256'][c] for c in corruptions})
    missing = [n for n in expected if not (repo / n).is_file()]
    clean = ['data_batch_' + str(i) for i in range(1, 6)] + ['test_batch', 'batches.meta']
    missing += ['data/cifar-10-batches-py/' + n for n in clean if not (repo/'data/cifar-10-batches-py'/n).is_file()]
    if missing:raise ValueError('缺少输入：' + ', '.join(missing))
    bad = [n for n, checksum in expected.items() if digest(repo / n) != checksum]
    if bad:raise ValueError('摘要不符：' + ', '.join(bad))
    return dict(source=proof,checked_hashes=len(expected),corruptions=list(corruptions),
                clean_files='仅确认存在；干净CIFAR完整性由官方加载器检查',model_executed=False)


def main():
    """输出核验结果；默认定位仓库旁的独立锁定工作树，不依赖服务器绝对路径。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=DEFAULT_RUNTIME)
    parser.add_argument('--corruptions',nargs='+')
    args=parser.parse_args()
    try:print(json.dumps(check_inputs(args.repo,args.corruptions),ensure_ascii=False,indent=2))
    except (ValueError,OSError) as error:parser.exit(2,f'输入核验失败：{error}\n')


if __name__=='__main__':
    main()
