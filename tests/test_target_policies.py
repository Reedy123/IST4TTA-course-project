"""合成概率验证视图索引、门控边界、硬化目标及随机状态不变。"""
import random
import unittest
import numpy as np
from experiments.target_policies import hard_only, selective_plca


class TargetTests(unittest.TestCase):
    """只依赖概率和提议，不存在真实标签输入接口。"""
    def fixture(self, strong=False, majority=6):
        """构造同图多数视图预测0、少数预测1、PLCA全部提议0。"""
        p=np.array([[.9,.1]]*majority+([[.05,.95]] if strong else [[.3,.7]])*(8-majority),dtype=np.float32)
        q=np.tile([1.,0.],(8,1));return p,q

    def test_hardening(self):
        """每视图独立argmax，保持矩阵形状与float32。"""
        p,_=self.fixture();t=hard_only(p);self.assertEqual(t.shape,(8,2));self.assertEqual(t.dtype,np.float32)
        np.testing.assert_array_equal(t.argmax(-1),p.argmax(-1))

    def test_accept(self):
        """低置信冲突被6/8多数支持时接受。"""
        p,q=self.fixture();t,r=selective_plca(p,q)
        self.assertEqual(r['plca_proposed_changes'],2);self.assertEqual(r['accepted_changes'],2)
        self.assertEqual(r['accepted_view_indices'],[6,7]);np.testing.assert_array_equal(t,q)

    def test_protect_confident(self):
        """高置信原预测冲突受到保护。"""
        p,q=self.fixture(True);t,r=selective_plca(p,q)
        self.assertEqual(r['accepted_changes'],0);np.testing.assert_array_equal(t,hard_only(p))

    def test_reject_unsupported(self):
        """5/8支持不足时拒绝提议。"""
        p,q=self.fixture(majority=5);t,r=selective_plca(p,q)
        self.assertEqual(r['accepted_changes'],0);np.testing.assert_array_equal(t,hard_only(p))

    def test_no_conflict(self):
        """相同标签不算作纠错或门控接受。"""
        p,_=self.fixture();t,r=selective_plca(p,hard_only(p))
        self.assertEqual(r['plca_proposed_changes'],0);self.assertEqual(r['accepted_changes'],0)

    def test_group_index(self):
        """相邻原图具有相反多数类，支持度不会串组。"""
        p,q=self.fixture();p2=p[:,::-1].copy();q2=q[:,::-1].copy()
        target,r=selective_plca(np.concatenate([p,p2]),np.concatenate([q,q2]))
        self.assertEqual(r['accepted_view_indices'],[6,7,14,15]);np.testing.assert_array_equal(target,np.concatenate([q,q2]))

    def test_probability_boundary(self):
        """置信度恰等于0.8按预先规定规则拒绝。"""
        p,q=self.fixture();p[6:]=[.2,.8];_,r=selective_plca(p,q)
        self.assertEqual(r['accepted_changes'],0)

    def test_immutable_rng(self):
        """计算不改变输入和全局Python/NumPy随机源。"""
        p,q=self.fixture();a=p.copy();b=q.copy();py=random.getstate();nr=np.random.get_state()
        selective_plca(p,q);hard_only(p)
        np.testing.assert_array_equal(p,a);np.testing.assert_array_equal(q,b);self.assertEqual(py,random.getstate())
        np.testing.assert_array_equal(nr[1],np.random.get_state()[1]);self.assertEqual(nr[2:],np.random.get_state()[2:])

    def test_partial_batch_and_shapes(self):
        """单原图不满批次可用，残缺视图组拒绝。"""
        p,q=self.fixture();self.assertEqual(selective_plca(p,q)[0].shape,(8,2))
        with self.assertRaises(AssertionError):selective_plca(p[:-1],q[:-1])


if __name__=='__main__':
    unittest.main()
