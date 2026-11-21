"""分阶段只读验证环境、权重和数据；不执行模型参数更新。"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

COMMIT = "03ba8e716f9bb6e0a586718f8836362b7090692b"
SHA = "a39f020c48b3bad58df794b5d3a286724f00b9f026ee22d0e6d1710748599e82"


def check_repo(repo):
    """核对官方版本、文件完整性和工作区状态。"""
    subprocess.run(["git", "merge-base", "--is-ancestor", COMMIT, "HEAD"], cwd=repo, check=True)
    subprocess.run(["git", "diff", "--exit-code", COMMIT, "--", "model", "dataset", "utils", "online.py", "online_clip.py", "scripts", "requirements.txt", "LICENSE", "README.md"], cwd=repo, check=True)
    assert not subprocess.check_output(["git", "status", "--porcelain", "--", "model", "dataset", "utils", "online.py", "online_clip.py", "scripts", "requirements.txt", "LICENSE", "README.md"], cwd=repo, text=True).strip(), "官方算法文件存在改动"
    sys.path.insert(0, str(repo))
    os.chdir(repo)


def gpu_pair(order):
    """在独立进程中验证两种导入顺序及 GPU 动态库共存。"""
    code = "import " + order + "\n" + """
import numpy as np
assert torch.cuda.is_available()
x = torch.randn(32, 32, device='cuda')
assert torch.isfinite(x @ x.T).all()
assert faiss.get_num_gpus() >= 1
res = faiss.StandardGpuResources()
index = faiss.GpuIndexFlatL2(res, 4)
a = np.eye(4, dtype='float32')
index.add(a)
d, i = index.search(a, 1)
assert np.array_equal(i[:, 0], np.arange(4))
torch.cuda.synchronize()
print('GPU 联合验证通过')
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def environment():
    """验证版本、GPU FAISS、半精度稀疏标签传播和单进程 NCCL。"""
    import importlib.metadata as metadata
    expected = {'torch': '1.13.1+cu117', 'torchvision': '0.14.1+cu117',
                'numpy': '1.23.5', 'scipy': '1.10.1', 'timm': '0.6.13'}
    assert sys.version_info[:2] == (3, 10)
    for name, version in expected.items():
        actual = metadata.version(name)
        print(name, actual)
        assert actual == version
    gpu_pair('torch, faiss')
    gpu_pair('faiss, torch')
    import torch
    import torch.distributed as dist
    import faiss
    import numpy as np
    from scipy.sparse import eye
    from scipy.sparse.linalg import cg
    import dataset
    import utils
    from utils.pseudo_labeling import PLCA
    from types import SimpleNamespace
    assert faiss.__version__ == '1.7.4'
    solution, info = cg(eye(4, format='csr'), np.ones(4), tol=1e-6)
    assert info == 0 and np.allclose(solution, 1)
    print(torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda)
    with tempfile.TemporaryDirectory() as temp:
        dist.init_process_group('nccl', init_method='file://' + temp + '/rendezvous', rank=0, world_size=1)
        try:
            value = torch.ones(1, device='cuda')
            dist.all_reduce(value)
            dist.barrier()
            rng = np.random.RandomState(42)
            sample = SimpleNamespace(features=list(rng.randn(80, 128).astype('float32')),
                                     pseudo_labels=list(np.ones((80, 10), dtype='float32') / 10))
            sample.set_pseudo_labels = lambda indexes, labels: None
            PLCA(sample, K=50, gamma=3, mode='l2', repeat=1, return_dst=True, verbose=False)
        finally:
            dist.destroy_process_group()
    print('环境验证通过；这里只测试合成数据，没有模型训练')


def weights(repo):
    """核对权重摘要、严格加载及推理输出，不更新模型。"""
    import torch
    from model import load_model
    path = repo / 'data/ModelZoo/Hendrycks2020AugMixWRN.pt'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SHA
    model = load_model('WRN-40-2', pretrained=True, num_classes=10, device='cuda')
    model.eval()
    with torch.no_grad():
        logits, features = model(torch.zeros(2, 3, 32, 32, device='cuda'))
    assert logits.shape == (2, 10) and features.shape == (2, 128)
    assert torch.isfinite(logits).all()
    print('模型权重验证通过')


def data(repo):
    """先禁止缺失数据触发下载，再核对官方加载器的第五级切片。"""
    import numpy as np
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor
    from dataset import load_dataset
    root = repo / 'data'
    # download=False 确保本验证过程不联网下载。
    clean = CIFAR10(str(root), train=False, download=False)
    raw = np.load(root / 'CIFAR-10-C/gaussian_noise.npy', mmap_mode='r')
    assert raw.shape == (50000, 32, 32, 3) and raw.dtype == np.uint8
    target = load_dataset('cifar10-c', str(root), ['gaussian_noise', 5], ToTensor())
    assert len(target) == 10000 and list(target.targets) == list(clean.targets)
    assert np.array_equal(target.data, raw[40000:50000])
    assert target[0][0].shape == (3, 32, 32)
    print('数据验证通过；标签来自原版 CIFAR-10，未使用 labels.npy')


def main():
    """根据阶段参数运行指定检查，失败时返回非零状态。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['env', 'weights', 'data'])
    parser.add_argument('--repo', required=True, type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    check_repo(repo)
    if args.stage == 'env':
        environment()
    elif args.stage == 'weights':
        weights(repo)
    else:
        data(repo)


if __name__ == '__main__':
    # 固定使用持久化环境，避免误用系统或 py38。
    target = Path(os.environ.get("IST_ENV_PREFIX", str(Path.home() / "work/envs/ist310"))) / "bin/python"
    assert target.is_file(), f"持久化环境不存在：{target}"
    if Path(sys.prefix).resolve() != target.parent.parent.resolve():
        os.execv(str(target), [str(target), str(Path(__file__).resolve()), *sys.argv[1:]])
    main()
