"""历史读取开关及只读诊断的确定性测试，不执行GPU实验。"""
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace
import json
import unittest
import numpy as np
import torch
from experiments.history import select_history, record_quality


class HistoryTests(unittest.TestCase):
    """保持原版对象与状态，并严格限制no-history生效阶段。"""
    def test_selection(self):
        """原版返回相同对象，适应禁用历史但最终评估不改变。"""
        features=[1,2];labels=[3,4];bank=SimpleNamespace(get_all=lambda:(features,labels))
        a,b=select_history(bank,True,1,'original');self.assertIs(a,features);self.assertIs(b,labels)
        self.assertEqual(select_history(bank,True,1,'no-history'),([],[]))
        self.assertIs(select_history(bank,True,0,'no-history')[0],features)
        self.assertEqual(select_history(bank,False,0,'original'),([],[]))

    def test_no_read(self):
        """no-history适应时连get_all也不调用。"""
        def fail():
            """如果意外读取历史，明确失败。"""
            raise AssertionError('历史被读取')
        self.assertEqual(select_history(SimpleNamespace(get_all=fail),True,1,'no-history'),([],[]))

    def test_quality_readonly(self):
        """诊断计数正确，不修改标签、输入和全局随机状态。"""
        soft=np.array([[.9,.1],[.8,.2],[.6,.4],[.7,.3]])
        corrected=np.array([[1,0],[1,0],[0,1],[0,1]])
        dataset=SimpleNamespace(labels=[0,1],get_soft_labels=lambda:soft,get_pseudo_labels=lambda:corrected)
        a=soft.copy();b=corrected.copy();py=random.getstate();nr=np.random.get_state();tr=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'q.jsonl';record_quality(dataset,([],[]),path,0,2);r=json.loads(path.read_text())
        self.assertEqual(r['raw_group_correct'],1);self.assertEqual(r['corrected_group_correct'],2)
        self.assertEqual(r['group_wrong_to_right'],1);self.assertEqual(r['history_entries_read'],0)
        self.assertTrue(np.array_equal(a,soft) and np.array_equal(b,corrected))
        self.assertEqual(py,random.getstate());self.assertTrue(torch.equal(tr,torch.get_rng_state()))
        self.assertTrue(np.array_equal(nr[1],np.random.get_state()[1]));self.assertEqual(nr[2:],np.random.get_state()[2:])


if __name__=='__main__':
    unittest.main()
