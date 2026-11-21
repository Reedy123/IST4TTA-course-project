"""验证分组索引、筛选模式、空入库、FIFO 和随机数不变性。"""
import unittest
import numpy as np
import torch
from experiments.admission import Admission
from utils.unlabeled_dataset import MemoryBank


class AdmissionTests(unittest.TestCase):
    """对原始记忆库执行确定性策略测试。"""
    def fixture(self, views=8):
        """构造高置信一致、低置信一致、高置信分歧三类原图。"""
        p = np.array([[[.9,.1]]*views, [[.55,.45]]*views,
                      [[.9,.1]]*(views//2)+[[.1,.9]]*(views-views//2)])
        features = list(range(3*views))
        labels = [np.array([1.,0.]) for _ in features]
        return features, labels, p.reshape(-1,2)

    def test_modes(self):
        """四模式选择预期原图，保持视图顺序。"""
        for mode, groups in [('original',3),('confidence',1),('consistency',2),('combined',1)]:
            f,l,p=self.fixture();b=MemoryBank(100);r=Admission(mode).update(b,f,l,p,8)
            self.assertEqual(r['accepted'],groups)
            self.assertEqual(b.features,list(range(groups*8)))

    def test_empty_and_zero_capacity(self):
        """无样本通过和零容量不会伪造入库。"""
        f,l,p=self.fixture();b=MemoryBank(100)
        self.assertEqual(Admission('combined',1,1).update(b,f,l,p,8)['bank_size'],0)
        b=MemoryBank(0);self.assertEqual(Admission().update(b,f,l,p,8)['bank_size'],0)

    def test_fifo_and_age(self):
        """奇数容量、后续空入库的年龄与原 FIFO 一致。"""
        f,l,p=self.fixture();b=MemoryBank(13);a=Admission('combined')
        a.update(b,f,l,p,8);a.update(b,f,l,p,8)
        self.assertEqual(b.features,list(range(3,8))+list(range(8)))
        a.confidence=1;r=a.update(b,f,l,p,8)
        self.assertEqual(r['age_max'],2);self.assertEqual(r['bank_size'],13)

    def test_varied_views_and_batches(self):
        """不同视图数和不满批次仍保持分组。"""
        for v in [1,2,8]:
            f,l,p=self.fixture(v);b=MemoryBank(None);Admission().update(b,f,l,p,v)
            self.assertEqual(len(b.features),3*v)

    def test_original_rng_and_objects(self):
        """original 与原函数的内容、对象及随机数状态一致。"""
        f,l,p=self.fixture();a=MemoryBank(13);b=MemoryBank(13)
        tr=torch.get_rng_state().clone();nr=np.random.get_state()
        Admission().update(a,f,l,p,8);b.update(f,l)
        self.assertEqual(a.features,b.features);self.assertIs(a.labels[0],b.labels[0])
        self.assertTrue(torch.equal(tr,torch.get_rng_state()))
        self.assertTrue(np.array_equal(nr[1],np.random.get_state()[1]))

    def test_evaluation_passthrough(self):
        """评估阶段不使用筛选规则。"""
        f,l,p=self.fixture();b=MemoryBank(100)
        Admission('combined',1,1).update(b,f,l,p,8,adapt=False)
        self.assertEqual(b.features,f)

    def test_noncontiguous_partial_batch(self):
        """不满批且首组拒绝时，保留后组原索引与 PLCA 标签对象。"""
        f,l,p=self.fixture()
        p=p.reshape(3,8,2)[[1,0]].reshape(16,2)
        features=list(range(16));labels=[np.array([i,16-i]) for i in range(16)]
        bank=MemoryBank(100)
        Admission('combined').update(bank,features,labels,p,8)
        self.assertEqual(bank.features,list(range(8,16)))
        self.assertIs(bank.labels[0],labels[8])

    def test_invalid_shape(self):
        """不完整视图组必须明确失败。"""
        f,l,p=self.fixture()
        with self.assertRaises(AssertionError):Admission().update(MemoryBank(),f[:-1],l[:-1],p,8)


if __name__=='__main__':
    unittest.main()
