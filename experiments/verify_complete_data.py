"""核验新增扰动的传输完整性、标签顺序及官方加载器，不覆盖已有文件。"""
import argparse
import json
import os
from pathlib import Path
import pickle
import numpy as np
from .complete_protocol import CORRUPTIONS
from .final_protocol import digest


def verify(repo, incoming, report):
    """先验证全部SHA256再硬链接到数据目录，原上传文件也保留。"""
    manifest = json.loads((incoming/'manifest.json').read_text())
    old=json.loads((Path.home()/'work/ist-final-stage1-data-verification.json').read_text())
    expected={r['corruption']:r['sha256'] for r in old['official_loader']}
    expected.update({Path(r['file']).stem:r['sha256'] for r in manifest['files']})
    assert set(expected)==set(CORRUPTIONS)
    destination = repo/'data/CIFAR-10-C'
    assert destination.is_dir()
    for r in manifest['files']:
        src = incoming/r['file']
        assert src.stat().st_size == r['bytes'] and digest(src) == r['sha256']
        a = np.load(src, mmap_mode='r', allow_pickle=False)
        assert a.shape == (50000,32,32,3) and a.dtype == np.uint8
        target = destination/r['file']
        if target.exists():
            assert digest(target) == r['sha256'], '已有文件不一致，拒绝覆盖'
    with (repo/'data/cifar-10-batches-py/test_batch').open('rb') as f:
        truth = np.asarray(pickle.load(f, encoding='bytes')[b'labels'])
    labels = np.load(destination/'labels.npy', allow_pickle=False)
    assert digest(destination/'labels.npy') == manifest['labels_sha256']
    assert np.array_equal(labels, truth if len(labels)==10000 else np.tile(truth,5))
    for r in manifest['files']:
        target = destination/r['file']
        if not target.exists():
            os.link(incoming/r['file'], target)
    from dataset import load_dataset, get_transforms
    loaded = []
    for name in CORRUPTIONS:
        assert digest(destination/(name+'.npy'))==expected[name], '数据摘要不符：'+name
        dataset = load_dataset('cifar10-c', str(repo/'data'), [name,5], preprocess=get_transforms('cifar10-c')[1])
        assert len(dataset) == 10000
        assert np.array_equal(dataset.targets, truth)
        assert dataset[0][0].shape == (3,32,32)
        loaded.append(dict(corruption=name, images=len(dataset), sha256=digest(destination/(name+'.npy'))))
    result = dict(transfer_sha256_verified=True, labels_match_clean=True, official_loader=loaded, manifest=manifest)
    report.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--incoming', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    verify(Path(__file__).resolve().parents[1], args.incoming, args.report)
