"""独立配对评估：固定共享视图，比较原始与 IST 权重，不更新参数或 BN。"""
import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def file_hash(path):
    """计算文件摘要以记录来源。"""
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def state_hash(model, bn_only=False):
    """对模型参数或 BN 缓冲区逐字节生成摘要。"""
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if bn_only and not any(k in name for k in ['running_mean', 'running_var', 'num_batches_tracked']):
            continue
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class SharedViews(Dataset):
    """按原始索引固定增强种子；每份张量由两个检查点共享。"""
    def __init__(self, root, seed, train_transform, test_transform, corruption="gaussian_noise"):
        """载入官方数据并固定全部一万张样本。"""
        from torchvision.datasets import CIFAR10
        clean = CIFAR10(str(root), train=False, download=False)
        self.labels = clean.targets
        self.images = np.load(root / 'CIFAR-10-C' / (corruption + '.npy'), mmap_mode='r')[40000:50000]
        assert self.images.shape == (10000, 32, 32, 3) and self.images.dtype == np.uint8
        self.seed, self.train_transform, self.test_transform = seed, train_transform, test_transform

    def __len__(self):
        """返回完整测试样本数。"""
        return 10000

    def __getitem__(self, index):
        """产生原图与八个随机增强，不依赖加载进程的随机状态。"""
        image = Image.fromarray(self.images[index])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + index)
            views = torch.stack([self.train_transform(image) for _ in range(8)])
        return index, self.labels[index], self.test_transform(image), views


def predict(model, inputs):
    """按官方批大小前向；返回概率，不创建梯度。"""
    outputs = []
    for chunk in inputs.split(128):
        logits, _ = model(chunk.cuda(non_blocking=True))
        outputs.append(logits.softmax(-1).cpu())
    return torch.cat(outputs)


def main():
    """执行两检查点、两口径评估并保存配置、预测及状态校验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corruption', choices=['gaussian_noise','motion_blur','fog','jpeg_compression'], required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--adapted', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    repo, adapted, out = args.repo.resolve(), args.adapted.resolve(), args.out.resolve()
    out.mkdir(exist_ok=False)
    sys.path.insert(0, str(repo))
    from dataset import get_transforms
    from model import load_model
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    source = repo / 'data/ModelZoo/Hendrycks2020AugMixWRN.pt'
    assert file_hash(source) == 'a39f020c48b3bad58df794b5d3a286724f00b9f026ee22d0e6d1710748599e82'
    models = {}
    for name, path in [('source', source), ('ist', adapted)]:
        model = load_model('WRN-40-2', pretrained=False, num_classes=10, device='cuda')
        model.load_state_dict(torch.load(path, map_location='cpu'), strict=True)
        model.eval(); model.requires_grad_(False)
        models[name] = model
    before = {name: {'state': state_hash(model), 'bn': state_hash(model, True)} for name, model in models.items()}
    train, test = get_transforms('cifar10-c')
    loader = DataLoader(SharedViews(repo / 'data', seed, train, test, args.corruption), batch_size=128, shuffle=False, num_workers=2)
    config = dict(corruption=args.corruption, level=5, seed=seed, augmentation_seed='42 + original_index', views=8, batch_size=128,
                  order='original 0..9999', preprocessing=dict(train=str(train), test=str(test)),
                  aggregation='argmax over 8*10 of softmax_probability squared; no averaging, no PLCA',
                  differences='Fixed original order and per-sample RNG; no DDP/shuffled RNG history; eight random views, no mandatory original view.',
                  source_sha256=file_hash(source), adapted_sha256=file_hash(adapted),
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip())
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    counts = {name: {'single': 0, 'eight': 0} for name in models}
    input_hash = hashlib.sha256(); seen = 0
    torch.cuda.synchronize(); start = time.monotonic()
    with (out / 'predictions.csv').open('w') as f, torch.inference_mode():
        writer = csv.writer(f); writer.writerow(['index','label','source_single','source_eight','ist_single','ist_eight'])
        for indices, labels, originals, views in loader:
            assert indices.tolist() == list(range(seen, seen + len(labels)))
            input_hash.update(originals.numpy().tobytes()); input_hash.update(views.numpy().tobytes())
            predictions = {}
            for name, model in models.items():
                assert not model.training and all(not m.training for m in model.modules())
                single = predict(model, originals).argmax(-1)
                probabilities = predict(model, views.flatten(0,1)).reshape(len(labels), 8, 10)
                eight = probabilities.square().flatten(1).argmax(-1) % 10
                predictions[name] = (single, eight)
                counts[name]['single'] += (single == labels).sum().item()
                counts[name]['eight'] += (eight == labels).sum().item()
            for k in range(len(labels)):
                writer.writerow([indices[k].item(), labels[k].item(), *[predictions[n][j][k].item() for n in ['source','ist'] for j in [0,1]]])
            seen += len(labels)
            print(f'Progress {seen}/10000 counts={counts}', flush=True)
    torch.cuda.synchronize()
    after = {name: {'state': state_hash(model), 'bn': state_hash(model, True)} for name, model in models.items()}
    assert seen == 10000 and before == after, '参数或 BN 缓冲区改变'
    results = dict(count=seen, accuracy={n: {k:v/seen for k,v in c.items()} for n,c in counts.items()},
                   before=before, after=after, state_unchanged=True, input_sha256=input_hash.hexdigest(),
                   elapsed_seconds=time.monotonic()-start)
    results['gain_percentage_points'] = {k:100*(results['accuracy']['ist'][k]-results['accuracy']['source'][k]) for k in ['single','eight']}
    (out / 'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2),flush=True)


if __name__ == '__main__':
    main()
