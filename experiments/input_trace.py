"""仅记录训练输入及随机状态摘要，不生成或改变随机数。"""
import hashlib
import json
import pickle
import random
from pathlib import Path
import numpy as np
import torch


def trace_batch(batch, dataset, step, destination):
    """检查视图张量顺序并记录输入及随机状态指纹。"""
    original = hashlib.sha256()
    for tensor in batch:
        original.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    augmented = hashlib.sha256()
    for image in dataset._samples:
        augmented.update(image.detach().cpu().contiguous().numpy().tobytes())
    rng = hashlib.sha256()
    rng.update(torch.get_rng_state().numpy().tobytes())
    for state in torch.cuda.get_rng_state_all():
        rng.update(state.cpu().numpy().tobytes())
    rng.update(pickle.dumps(random.getstate()))
    rng.update(pickle.dumps(np.random.get_state()))
    record = dict(batch=step, count=len(batch[0]), input_sha256=original.hexdigest(),
                  augmented_sha256=augmented.hexdigest(), rng_sha256=rng.hexdigest())
    with Path(destination).open('a') as f:
        f.write(json.dumps(record)+'\n')
