"""校验无传播入口保持软目标、损失形式及随机状态。"""
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F
from experiments.no_plca import raw_prediction_labels


class NoPLCATests(unittest.TestCase):
    """无需GPU验证软标签及分支接口。"""
    def test_bypass_and_shape(self):
        """适应repeat被覆盖为0，其余参数与概率对象保持不变。"""
        p=np.full((16,10),.1,dtype=np.float32)
        item=SimpleNamespace(soft_labels=p,pseudo_labels=list(p))
        tr=torch.get_rng_state().clone();nr=np.random.get_state();pr=random.getstate()
        with patch('experiments.no_plca.robust_PLCA',return_value=item) as mock:
            result=raw_prediction_labels('samples','classes',repeat=1,K=50,gamma=3,return_dst=True)
        self.assertIs(result,item);self.assertEqual(mock.call_args.kwargs['repeat'],0)
        self.assertEqual(mock.call_args.kwargs['K'],50)
        self.assertTrue(torch.equal(tr,torch.get_rng_state()));self.assertEqual(pr,random.getstate())
        self.assertTrue(np.array_equal(nr[1],np.random.get_state()[1]))

    def test_loss_interface(self):
        """原交叉熵和KL公式接受同形状软目标，保持有效梯度。"""
        logits=torch.tensor([[1.,2.,3.],[3.,2.,1.]],requires_grad=True)
        p=torch.tensor([[.2,.3,.5],[.5,.2,.3]])
        loss=(-p*F.log_softmax(logits,dim=-1)).sum(-1).mean()
        loss+=F.kl_div(F.log_softmax(logits,dim=-1),p,reduction='batchmean')
        loss.backward();self.assertTrue(torch.isfinite(loss))
        self.assertEqual(logits.grad.shape,p.shape);self.assertGreater(logits.grad.abs().sum().item(),0)

    def test_no_hardening(self):
        """意外硬化或形状错位必须被拒绝。"""
        p=np.full((8,10),.1,dtype=np.float32);q=np.zeros((8,10));q[:,0]=1
        with patch('experiments.no_plca.robust_PLCA',return_value=SimpleNamespace(soft_labels=p,pseudo_labels=q)):
            with self.assertRaises(AssertionError):raw_prediction_labels(return_dst=True)


if __name__=='__main__':
    unittest.main()
