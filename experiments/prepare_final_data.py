"""只从已下载的官方完整包提取新增三种扰动，不覆盖原始数据。"""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
import hashlib
import numpy as np
from .final_protocol import CORRUPTIONS, digest


def prepare(base, out):
    """重新核对官方MD5并验证逐文件提取字节、形状、等级与标签。"""
    archive = base/'downloads/CIFAR-10-C.tar'
    h = hashlib.md5()
    with archive.open('rb') as f:
        for b in iter(lambda: f.read(8388608), b''):
            h.update(b)
    assert archive.stat().st_size == 2918471680
    assert h.hexdigest() == '56bf5dcef84df0e2308c6dcbcbbd8499'
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    with tarfile.open(archive) as tar:
        for name in CORRUPTIONS[1:]:
            member = tar.getmember('CIFAR-10-C/'+name+'.npy')
            assert member.isfile() and member.size == 153600128
            target = out/(name+'.npy')
            with tar.extractfile(member) as src, target.open('xb') as dest:
                shutil.copyfileobj(src, dest)
            with tar.extractfile(member) as src:
                expected = hashlib.sha256(src.read()).hexdigest()
            assert digest(target) == expected
            a = np.load(target, mmap_mode='r', allow_pickle=False)
            assert a.shape == (50000, 32, 32, 3) and a.dtype == np.uint8
            rows.append(dict(file=target.name, bytes=target.stat().st_size, sha256=expected,
                             shape=list(a.shape), dtype=str(a.dtype), level5=[40000, 50000]))
    labels = np.load(base/'data/CIFAR-10-C/labels.npy', allow_pickle=False)
    assert labels.shape in [(10000,), (50000,)]
    result = dict(archive_md5=h.hexdigest(), archive_sha256=digest(archive), files=rows,
                  labels_sha256=digest(base/'data/CIFAR-10-C/labels.npy'))
    (out/'manifest.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.base, args.out)
