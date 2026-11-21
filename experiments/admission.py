"""按原图成组筛选历史记忆；不修改训练标签或消耗随机数。"""
import json
from pathlib import Path
import numpy as np


class Admission:
    """记录入库策略、FIFO 对齐的年龄及类别分布。"""
    def __init__(self, mode='original', confidence=0.8, consistency=0.75, log=None):
        """初始化固定阈值，不访问真实标签。"""
        assert mode in ['original', 'confidence', 'consistency', 'combined']
        assert 0 <= confidence <= 1 and 0 <= consistency <= 1
        self.mode, self.confidence, self.consistency = mode, confidence, consistency
        self.log = Path(log) if log else None
        self.births = []
        self.step = 0
        self.total = self.accepted = 0

    def update(self, bank, features, labels, probabilities, views, adapt=True):
        """仅适应阶段筛选，保留所选条目的原有对象及原版 FIFO 语义。"""
        assert len(features) == len(labels) and views > 0 and len(features) % views == 0
        groups = len(features) // views
        if not adapt:
            bank.update(features, labels)
            return None
        probs = np.asarray(probabilities).reshape(groups, views, -1)
        assert np.isfinite(probs).all()
        votes = probs.argmax(-1)
        counts = np.stack([(votes == c).sum(1) for c in range(probs.shape[-1])], axis=1)
        majority = counts.argmax(1)
        agree = counts.max(1) / views
        confidence = probs[np.arange(groups)[:, None], np.arange(views)[None, :], majority[:, None]].mean(1)
        keep = np.ones(groups, dtype=bool)
        if self.mode in ['confidence', 'combined']:
            keep &= confidence >= self.confidence
        if self.mode in ['consistency', 'combined']:
            keep &= agree >= self.consistency
        # 两个判据在同一轨迹上的四分区，避免把不同模型轨迹混为同一集合。
        confidence_mask = confidence >= self.confidence
        consistency_mask = agree >= self.consistency
        indices = np.flatnonzero(np.repeat(keep, views))
        # original 直接调用原函数并传递原列表，避免类型和顺序变化。
        selected_features = features if self.mode == 'original' else [features[i] for i in indices]
        selected_labels = labels if self.mode == 'original' else [labels[i] for i in indices]
        bank.update(selected_features, selected_labels)
        self.births.extend([self.step] * len(indices))
        capacity = bank.max_len
        if capacity is not None and len(self.births) > capacity:
            self.births = self.births[-capacity:] if capacity > 0 else []
        assert len(self.births) == bank.size()
        self.total += groups
        self.accepted += int(keep.sum())
        ages = np.asarray([self.step - b for b in self.births])
        classes = np.asarray(bank.labels).argmax(1) if bank.size() else np.array([], dtype=int)
        record = dict(step=self.step, mode=self.mode, groups=groups, accepted=int(keep.sum()),
                      admission_rate=float(keep.mean()) if groups else 0., cumulative_rate=self.accepted / self.total,
                      bank_size=bank.size(), class_counts=np.bincount(classes, minlength=probs.shape[-1]).tolist(),
                      age_mean=float(ages.mean()) if len(ages) else 0., age_max=int(ages.max()) if len(ages) else 0,
                      age_unit='adaptation batches', confidence_mean=float(confidence.mean()), consistency_mean=float(agree.mean()))
        record.update(confidence_only=int((confidence_mask & ~consistency_mask).sum()),
                      consistency_only=int((~confidence_mask & consistency_mask).sum()),
                      both=int((confidence_mask & consistency_mask).sum()),
                      neither=int((~confidence_mask & ~consistency_mask).sum()),
                      stream_offset=self.total-groups,
                      accepted_groups=np.flatnonzero(keep).tolist(),
                      confidence_groups=np.flatnonzero(confidence_mask).tolist(),
                      consistency_groups=np.flatnonzero(consistency_mask).tolist())
        if self.log:
            with self.log.open('a') as f:
                f.write(json.dumps(record) + '\n')
        self.step += 1
        return record
