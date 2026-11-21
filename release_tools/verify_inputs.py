"""用CPU只读核对源权重、所选扰动SHA-256及干净CIFAR文件存在性。"""
import argparse
import json
from pathlib import Path
from locked_source import verify, digest, MANIFESTS


def main():
    """明确区分内容摘要核验与尚需官方加载器确认的标签/格式检查。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--corruptions', nargs='+', help='默认核对全部15种扰动')
    args = parser.parse_args()
    try:
        proof = verify(args.repo)
        protocol = json.loads((MANIFESTS / 'formal-protocol.json').read_text())
        corruptions = args.corruptions or protocol['corruptions']
        if set(corruptions) - set(protocol['corruptions']):
            raise ValueError('扰动名称不属于锁定协议')
        expected = {'data/ModelZoo/Hendrycks2020AugMixWRN.pt': protocol['source_weight_sha256']}
        expected.update({'data/CIFAR-10-C/' + c + '.npy': protocol['data_sha256'][c] for c in corruptions})
        missing = [n for n in expected if not (args.repo / n).is_file()]
        clean = ['data_batch_' + str(i) for i in range(1, 6)] + ['test_batch', 'batches.meta']
        missing += ['data/cifar-10-batches-py/' + n for n in clean
                    if not (args.repo / 'data/cifar-10-batches-py' / n).is_file()]
        if missing:
            raise ValueError('缺少输入：' + ', '.join(missing))
        bad = [n for n, checksum in expected.items() if digest(args.repo / n) != checksum]
        if bad:
            raise ValueError('摘要不符：' + ', '.join(bad))
        print(json.dumps(dict(source=proof, checked_hashes=len(expected), corruptions=corruptions,
                              clean_files='仅确认存在；完整性/标签关系仍由官方数据加载器和验证步骤检查',
                              model_executed=False), ensure_ascii=False, indent=2))
    except (ValueError, OSError) as error:
        parser.exit(2, f'输入核验失败：{error}\n')


if __name__ == '__main__':
    main()
