"""使用独立 PCG64 随机源，按既有 Confidence 逐批配额随机入库。"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .admission import Admission


class RandomAdmission(Admission):
    """只匹配历史入库数量，不读取标签正确性或根据准确率调整选择。"""
    def __init__(self, mode='random-admission', confidence=0.8, consistency=0.75,
                 log=None, schedule=None, random_seed=42):
        """固定配额文件和独立随机种子；不使用全局随机源。"""
        assert mode == 'random-admission' and schedule is not None
        super().__init__('original', confidence, consistency, log)
        self.mode = mode
        self.rng = np.random.Generator(np.random.PCG64(random_seed))
        path = Path(schedule)
        self.schedule_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        self.schedule = [json.loads(line) for line in path.read_text().splitlines()]
        assert all(row['step'] == i for i, row in enumerate(self.schedule))

    def update(self, bank, features, labels, probabilities, views, adapt=True):
        """先抽取原图组，再保留该组全部原特征与原 PLCA 标签。"""
        assert len(features) == len(labels) and views > 0 and len(features) % views == 0
        if not adapt:
            bank.update(features, labels)
            return None
        groups = len(features) // views
        assert self.step < len(self.schedule)
        expected = self.schedule[self.step]
        assert groups == expected['groups']
        accepted = int(expected['accepted'])
        assert 0 <= accepted <= groups
        # 排序恢复原输入顺序，避免额外引入 FIFO 顺序变化。
        chosen = np.sort(self.rng.choice(groups, size=accepted, replace=False))
        keep = np.zeros(groups, dtype=bool)
        keep[chosen] = True
        indices = np.flatnonzero(np.repeat(keep, views))
        bank.update([features[i] for i in indices], [labels[i] for i in indices])
        self.births.extend([self.step] * len(indices))
        capacity = bank.max_len
        if capacity is not None and len(self.births) > capacity:
            self.births = self.births[-capacity:] if capacity > 0 else []
        assert len(self.births) == bank.size()
        self.total += groups
        self.accepted += accepted
        ages = np.asarray([self.step - value for value in self.births])
        classes = np.asarray(bank.labels).argmax(1) if bank.size() else np.array([], dtype=int)
        num_classes = np.asarray(probabilities).shape[-1]
        record = dict(step=self.step, mode=self.mode, groups=groups, accepted=accepted,
                      admission_rate=accepted/groups, cumulative_rate=self.accepted/self.total,
                      bank_size=bank.size(), class_counts=np.bincount(classes, minlength=num_classes).tolist(),
                      age_mean=float(ages.mean()) if len(ages) else 0., age_max=int(ages.max()) if len(ages) else 0,
                      age_unit='adaptation batches', stream_offset=self.total-groups,
                      accepted_groups=chosen.tolist(), schedule_sha256=self.schedule_hash)
        if self.log:
            with self.log.open('a') as file:
                file.write(json.dumps(record)+'\n')
        self.step += 1
        return record
