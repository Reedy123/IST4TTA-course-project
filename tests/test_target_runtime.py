"""校验策略调用时机、真实标签隔离、特征及训练目标接口。"""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from experiments.target_runtime import make_target_policy


class FakeResult:
    """若策略试图读取真值则报错的最小返回对象。"""
    def __init__(self, propagated):
        """提供两个低置信冲突视图及原始特征对象。"""
        self.soft_labels=np.array([[.9,.1]]*6+[[.3,.7]]*2,dtype=np.float32)
        self.pseudo_labels=list(np.tile([1.,0.],(8,1))) if propagated else list(self.soft_labels)
        self.features=object()

    @property
    def labels(self):
        """禁止门控读取真实标签。"""
        raise AssertionError('策略不能读取真实标签')

    def set_pseudo_labels(self, indices, targets):
        """模拟官方索引写回。"""
        assert indices==list(range(8));self.pseudo_labels=list(targets)


class RuntimeTests(unittest.TestCase):
    """无需GPU检查回调分支与现有数据结构兼容。"""
    def test_hard_only(self):
        """原始软标签仅硬化，特征与soft_labels保持原对象。"""
        item=FakeResult(False);features=item.features;p=item.soft_labels;tr=torch.get_rng_state().clone()
        with patch('experiments.target_runtime.robust_PLCA',return_value=item) as mock:
            out=make_target_policy('hard-only',8)('samples',repeat=1,K=50,return_dst=True)
        self.assertEqual(mock.call_args.kwargs['repeat'],0);self.assertIs(out.features,features);self.assertIs(out.soft_labels,p)
        np.testing.assert_array_equal(np.asarray(out.pseudo_labels).argmax(-1),p.argmax(-1))
        self.assertTrue(torch.equal(tr,torch.get_rng_state()))

    def test_selective(self):
        """保留PLCA执行，门控不用真值即可接受两次提议。"""
        item=FakeResult(True)
        with patch('experiments.target_runtime.robust_PLCA',return_value=item) as mock:
            out=make_target_policy('selective-plca',8)('samples',repeat=1,K=50,return_dst=True)
        self.assertEqual(mock.call_args.kwargs['repeat'],1)
        np.testing.assert_array_equal(np.asarray(out.pseudo_labels).argmax(-1),np.zeros(8))

    def test_final_evaluation_unchanged(self):
        """repeat=0最终评估不能套用硬化或选择性目标。"""
        for mode in ['hard-only','selective-plca']:
            item=FakeResult(False);before=np.asarray(item.pseudo_labels).copy()
            with patch('experiments.target_runtime.robust_PLCA',return_value=item):
                out=make_target_policy(mode,8)('samples',repeat=0,return_dst=True)
            np.testing.assert_array_equal(before,out.pseudo_labels)


if __name__=='__main__':
    unittest.main()
