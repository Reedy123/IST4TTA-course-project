"""复用官方原始软预测路径，只跳过PLCA图传播，不创建新训练标签。"""
import numpy as np
from utils.pseudo_labeling import robust_PLCA


def raw_prediction_labels(*args, **kwargs):
    """关闭传播迭代，校验原始概率被原样保留为训练目标。"""
    kwargs = dict(kwargs)
    kwargs['repeat'] = 0
    result = robust_PLCA(*args, **kwargs)
    probabilities = np.asarray(result.soft_labels)
    targets = np.asarray(result.pseudo_labels)
    assert probabilities.ndim == 2 and targets.shape == probabilities.shape
    assert np.isfinite(targets).all() and (targets >= 0).all()
    assert np.allclose(targets.sum(-1), 1, atol=1e-5)
    assert np.array_equal(probabilities, targets), '关闭传播后标签必须等于原始softmax分布'
    return result
