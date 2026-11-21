"""检查等配额抽样、分组、随机源隔离和边界。"""
import json
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
import torch
from experiments.random_admission import RandomAdmission
from utils.unlabeled_dataset import MemoryBank


class RandomTests(unittest.TestCase):
    """独立抽样器不影响训练的全局随机状态。"""
    def run_case(self, counts, capacity=100):
        """构造八原图八视图的多批配额，并返回每批选择。"""
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'schedule.jsonl'
            path.write_text('\n'.join(json.dumps(dict(step=i,groups=8,accepted=k)) for i,k in enumerate(counts)))
            strategy=RandomAdmission(schedule=path,random_seed=42)
            bank=MemoryBank(capacity);records=[]
            features=list(range(64));labels=[np.array([1.,0.]) for _ in features]
            for count in counts:
                record=strategy.update(bank,features,labels,np.ones((64,2))*.5,8)
                self.assertEqual(record['accepted'],count)
                self.assertEqual(len(set(record['accepted_groups'])),count)
                records.append(record)
            return records,bank

    def test_group_mapping(self):
        """入库条目恰好是选中原图的八个视图且顺序不变。"""
        records,bank=self.run_case([3])
        self.assertEqual(bank.features,[v for g in records[0]['accepted_groups'] for v in range(g*8,(g+1)*8)])

    def test_reproducible(self):
        """同一个独立随机种子获得相同选择序列。"""
        a,_=self.run_case([3,4]);b,_=self.run_case([3,4])
        self.assertEqual([r['accepted_groups'] for r in a],[r['accepted_groups'] for r in b])

    def test_global_random_unchanged(self):
        """Python、NumPy 和 Torch 全局随机状态均保持原样。"""
        py=random.getstate();npstate=np.random.get_state();t=torch.get_rng_state().clone()
        self.run_case([0,3,8])
        self.assertEqual(py,random.getstate());self.assertTrue(torch.equal(t,torch.get_rng_state()))
        now=np.random.get_state();self.assertEqual(npstate[0],now[0]);self.assertTrue(np.array_equal(npstate[1],now[1]));self.assertEqual(npstate[2:],now[2:])

    def test_empty_full_and_fifo(self):
        """零配额、全接收、非整组容量沿用原 FIFO。"""
        rows,b=self.run_case([0,8],13)
        self.assertEqual(rows[0]['bank_size'],0);self.assertEqual(b.features,list(range(51,64)))
        _,b=self.run_case([8],0);self.assertEqual(b.size(),0)


if __name__=='__main__':
    unittest.main()
