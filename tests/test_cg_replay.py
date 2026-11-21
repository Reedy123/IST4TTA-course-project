"""固定图诊断的标签口径、暂时平台和CG回调编号测试。"""
import unittest
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import cg
from experiments.cg_replay import scores_to_labels,stability,metrics,section

class ReplayTests(unittest.TestCase):
    """不加载模型，不运行训练，测试诊断本身。"""
    def test_half_argmax(self):
        """官方半精度硬化可能与双精度argmax不同。"""
        scores=np.array([[1.,1.0001],[.1,.8]])
        self.assertEqual(scores.argmax(1)[0],1);self.assertEqual(scores_to_labels(scores)[0],0)
    def test_false_plateau(self):
        """短期稳定不等于未来不会翻转。"""
        path=np.zeros((10,2),int);path[6:,0]=1;r=stability(path)
        self.assertEqual(r['offline_plateau_candidate'],5);self.assertTrue(r['any_later_change_after_plateau']);self.assertEqual(r['first_permanent_match_to_final'],7)
    def test_permanent(self):
        """完全稳定与没有平台的轨迹均能正确处理。"""
        self.assertEqual(stability(np.zeros((10,2),int))['first_permanent_match_to_final'],1)
        path=np.arange(10)[:,None]%2;self.assertIsNone(stability(path)['offline_plateau_candidate'])
    def test_metrics(self):
        """原图八视图和逐视图计数分开，真值只进入评价。"""
        p=np.tile([.9,.1],(8,1));r=metrics(np.ones(8,int),p,np.array([1]))
        self.assertEqual(r['view_correct'],8);self.assertEqual(r['group_correct'],1);self.assertEqual(r['view_wrong_to_right'],8)
    def test_callback_numbering(self):
        """回调的第k个状态与原生maxiter=k严格相同。"""
        A=scipy.sparse.diags([1.,2.,4.,7.,11.]);b=np.ones(5);snapshots=[]
        x,info=cg(A,b,tol=1e-12,maxiter=3,callback=lambda x:snapshots.append(x.copy()))
        self.assertEqual(len(snapshots),3)
        for k in [1,2,3]:self.assertTrue(np.array_equal(cg(A,b,tol=1e-12,maxiter=k)[0],snapshots[k-1]))

if __name__=='__main__':unittest.main()
