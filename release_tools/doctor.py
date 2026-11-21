"""检查当前解释器、历史核心版本与GPU可用性；默认不执行GPU算子。"""
import argparse
import importlib.metadata as metadata
import json
import platform
import os
import shutil
import subprocess
import sys
from pathlib import Path

EXPECTED = {'torch': '1.13.1+cu117', 'torchvision': '0.14.1+cu117', 'numpy': '1.23.5',
            'scipy': '1.10.1', 'timm': '0.6.13', 'Pillow': '9.5.0', 'PyYAML': '6.0.1'}


def inspect_environment(require_gpu=False, device=0):
    """仅在显式要求GPU检查时导入torch/faiss并做小型联合运算。"""
    versions = {}
    for name in [*EXPECTED, 'faiss', 'faiss-gpu']:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    conda_versions = {}
    for path in (Path(sys.prefix)/'conda-meta').glob('*.json'):
        info = json.loads(path.read_text())
        if info.get('name') in ['mkl','cudatoolkit','faiss-gpu','libfaiss']:
            conda_versions[info['name']] = info.get('version')
    problems = [f'{n}: 需要{v}，当前{versions[n]}' for n, v in EXPECTED.items() if versions[n] != v]
    if sys.version_info[:3] != (3, 10, 13):
        problems.append('论文环境Python为3.10.13，当前解释器不同')
    for name, version in {'mkl':'2021.4.0','cudatoolkit':'11.4.1','faiss-gpu':'1.7.4','libfaiss':'1.7.4'}.items():
        if conda_versions.get(name) != version:
            problems.append(f'{name}: 需要{version}，conda记录为{conda_versions.get(name)}')
    result = {'conda_core_versions':conda_versions, 'python': sys.executable, 'python_version': platform.python_version(), 'platform': platform.platform(),
              'versions': versions, 'core_version_problems': problems, 'nvidia_smi': shutil.which('nvidia-smi'),
              'gpu_test': '未执行', 'paper_hardware_verified': False}
    if not require_gpu:
        return result
    if platform.system() != 'Linux' or not result['nvidia_smi'] or problems:
        raise ValueError('Linux/NVIDIA或核心版本检查未通过：' + json.dumps(result, ensure_ascii=False))
    from runtime_support import gpu_idle_info
    result['device_check'] = gpu_idle_info(device)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    import torch
    import faiss
    if faiss.__version__ != '1.7.4' or not torch.cuda.is_available() or faiss.get_num_gpus() < 1:
        raise ValueError('要求FAISS GPU1.7.4与可用CUDA设备，不能替换成CPU FAISS')
    # 环境诊断只分配微小测试张量，不训练模型；运行前仍应确认设备空闲。
    import numpy as np
    resources = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatL2(resources, 4)
    index.add(np.eye(4, dtype='float32'))
    _, neighbors = index.search(np.eye(4, dtype='float32'), 1)
    if not np.array_equal(neighbors[:, 0], np.arange(4)):
        raise ValueError('FAISS联合检查失败')
    tensor = torch.eye(4, device='cuda')
    if not torch.isfinite(tensor @ tensor).all():
        raise ValueError('PyTorch CUDA检查失败')
    result.update(gpu_test='通过联合微小算子检查（不是模型冒烟）', gpu=torch.cuda.get_device_name(0),
                  torch_cuda=torch.version.cuda, faiss_version=faiss.__version__,
                  paper_hardware_verified=torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3080')
    return result


def main():
    """输出可审阅诊断；GPU诊断失败时返回非零状态。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-gpu', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    try:
        print(json.dumps(inspect_environment(args.require_gpu, args.device), ensure_ascii=False, indent=2))
    except (ValueError, ImportError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, f'环境检查失败：{error}\n')


if __name__ == '__main__':
    main()
