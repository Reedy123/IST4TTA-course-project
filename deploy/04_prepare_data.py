"""从官方地址续传、校验并安全提取最小数据；不改写官方源码或原始压缩包。"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import tarfile

SOURCES = [
 ('cifar-10-python.tar.gz', 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', 170498071, 'c58f30108f718f92721af3b95e74349a'),
 ('CIFAR-10-C.tar', 'https://zenodo.org/api/records/2535967/files/CIFAR-10-C.tar/content', 2918471680, '56bf5dcef84df0e2308c6dcbcbbd8499'),
]


def digest(path, algorithm):
    """以固定内存计算文件摘要。"""
    result = hashlib.new(algorithm)
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def obtain(base, download):
    """仅在长度和官方 MD5 通过后，将部分文件标记为完整包。"""
    records = []
    for name, url, size, md5 in SOURCES:
        target = base / 'downloads' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(name + '.part')
        if not target.exists():
            if not download:
                raise FileNotFoundError(f'缺少完整包：{target}；可用 --download 续传')
            subprocess.run(['curl', '-L', '--fail', '--retry', '5', '--connect-timeout', '30',
                            '--continue-at', '-', url, '-o', str(part)], check=True)
            assert part.stat().st_size == size, f'{name} 长度不符'
            assert digest(part, 'md5') == md5, f'{name} MD5 不符，保留文件供排查'
            part.rename(target)
        assert target.stat().st_size == size
        assert digest(target, 'md5') == md5
        records.append(dict(file=name, bytes=size, md5=md5, sha256=digest(target, 'sha256'), source=url))
    return records


def extract(base):
    """仅复制白名单普通文件，不解压链接和任意路径。"""
    for name, _, _, _ in SOURCES:
        with tarfile.open(base / 'downloads' / name) as archive:
            for member in archive:
                pure = Path(member.name)
                clean = (name.startswith('cifar-10-python') and len(pure.parts) == 2
                         and pure.parts[0] == 'cifar-10-batches-py'
                         and pure.name in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4',
                                           'data_batch_5', 'test_batch', 'batches.meta', 'readme.html'])
                corrupt = member.name in ['CIFAR-10-C/gaussian_noise.npy', 'CIFAR-10-C/labels.npy']
                if not (member.isfile() and (clean or corrupt)):
                    continue
                destination = base / 'data' / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, destination.open('wb') as output:
                    shutil.copyfileobj(source, output)


def validate(base):
    """使用已有 NumPy 核对形状及五个等级的标签一致性。"""
    import numpy as np
    root = base / 'data'
    noise = np.load(root / 'CIFAR-10-C/gaussian_noise.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(root / 'CIFAR-10-C/labels.npy', allow_pickle=False)
    assert noise.shape == (50000, 32, 32, 3) and noise.dtype == np.uint8
    # 原版官方 pickle 只在压缩包 MD5 已核对后读取。
    with (root / 'cifar-10-batches-py/test_batch').open('rb') as source:
        clean = pickle.load(source, encoding='bytes')
    targets = np.asarray(clean[b'labels'])
    assert clean[b'data'].shape == (10000, 3072)
    assert targets.shape == (10000,) and targets.min() == 0 and targets.max() == 9
    assert labels.shape in [(10000,), (50000,)]
    expected = targets if labels.shape == (10000,) else np.tile(targets, 5)
    assert np.array_equal(labels, expected), '官方损坏标签与原版测试顺序不一致'
    assert noise[40000:50000].shape == (10000, 32, 32, 3)
    return [dict(file=str(path.relative_to(base)), bytes=path.stat().st_size,
                 sha256=digest(path, 'sha256')) for path in sorted(root.rglob('*')) if path.is_file()]


def main():
    """按下载、完整性校验、提取和语义检查顺序生成清单。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    base = args.base.resolve()
    records = obtain(base, args.download)
    extract(base)
    files = validate(base)
    (base / 'verified-manifest.json').write_text(json.dumps(dict(archives=records, files=files), indent=2))
    print('数据完整性、形状及标签验证通过')


if __name__ == '__main__':
    main()
