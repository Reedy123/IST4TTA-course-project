"""锁定正式实验配置，提供四模式映射和完整性检查。"""
import hashlib
import json
from pathlib import Path

MODES = {
    'Original': ('original', 'original'),
    'Original-Reuse': ('original', 'reuse-resources'),
    'No-history': ('no-history', 'original'),
    'Ours': ('no-history', 'reuse-resources'),
}
CORRUPTIONS = ['gaussian_noise', 'motion_blur', 'fog', 'jpeg_compression']
SOURCE_SHA = 'a39f020c48b3bad58df794b5d3a286724f00b9f026ee22d0e6d1710748599e82'


def digest(path):
    """流式计算摘要，不把大型数据文件整体读入内存。"""
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8388608), b''):
            h.update(b)
    return h.hexdigest()


def records(path):
    """读取逐批记录；缺失和损坏直接报错，不跳过。"""
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def command(repo, output, mode, corruption, batches=0):
    """生成固定参数命令，每组均以独立进程加载原始预训练模型。"""
    repo = Path(repo)
    assert mode in MODES and corruption in CORRUPTIONS
    return [str(Path.home()/'work/envs/ist310/bin/python'), '-m', 'torch.distributed.launch',
            '--nproc_per_node', '1', '--master_port', '16735', str(repo/'online_final.py'),
            '--mode', mode, '--profile_kind', 'normal', '--audit_faiss', '--skip_final_eval',
            '--max_batches', str(batches), '--dataset', 'cifar10-c', '--dataset_path', './data/',
            '--corruption', corruption, '--level', '5', '--model', 'WRN-40-2', '--pretrained',
            '--num_classes', '10', '--seed', '42', '--iters', '1', '--batch_size', '128',
            '--extend', '8', '--max_len', '10000', '--lr', '0.001', '--num_workers', '2',
            '--save_model', '--save_path', str(output)]


def exact_pair(a, b):
    """验证资源复用前后全部近邻、距离、伪标签、输入、随机状态与模型张量一致。"""
    import torch
    a, b = Path(a), Path(b)
    for name in ['input_trace.jsonl', 'faiss-audit.jsonl', 'pseudo-audit.jsonl',
                 'pseudo_quality.jsonl', 'admission.jsonl']:
        assert (a/name).read_bytes() == (b/name).read_bytes(), name
    x = torch.load(next(a.glob('*adapt.pth')), map_location='cpu')
    y = torch.load(next(b.glob('*adapt.pth')), map_location='cpu')
    assert x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x), '模型权重不一致'
    return dict(inputs_rng_exact=True, neighbors_distances_exact=True,
                pseudo_labels_exact=True, model_tensors_exact=True)
