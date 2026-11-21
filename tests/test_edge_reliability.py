"""验证留一视图公式、真实自环、传播结构及随机数隔离。"""
import ast
import copy
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
import torch
from experiments.edge_reliability import entropy,reliability,weights
from experiments.weighted_plca import EdgeRuntime,InsertWeights
from experiments.faiss_runtime import ReplaceCalls
from experiments.profiled_plca import instrument_tree,Strip
from experiments.profiling_runtime import profiler
from tests.test_faiss_runtime import Restore


class RemoveObservers(ast.NodeTransformer):
    """移除新观察和加权调用，验证官方计算未被改写。"""
    def visit_Assign(self,node):
        """删除唯一新增的relation赋值。"""
        self.generic_visit(node)
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='edge':return None
        return node
    def visit_Expr(self,node):
        """删除只读归一化图记录。"""
        self.generic_visit(node)
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='edge':return None
        return node


class EdgeTests(unittest.TestCase):
    """纯逻辑及CPU张量检查，不需要运行训练。"""
    def test_uniform(self):
        """均匀概率低置信、完全一致；联合权重下限为0.5。"""
        p=np.full((8,10),.1);c,a,r,h=weights(p,'edge-cv')
        self.assertTrue(np.allclose(c,0));self.assertTrue(np.allclose(a,1));self.assertTrue(np.allclose(h,.5))
    def test_one_hot_consensus(self):
        """两张不同原图分别一致时，各自可靠性均为1。"""
        p=np.array([[1.,0.]]*8+[[0.,1.]]*8);c,a=reliability(p)
        self.assertTrue(np.allclose(c,1));self.assertTrue(np.allclose(a,1))
    def test_leave_one_out(self):
        """孤立异常视图不能把自己的票计入参考。"""
        p=np.array([[0.,1.]]+[[1.,0.]]*7);c,a=reliability(p)
        self.assertAlmostEqual(a[0],0);self.assertTrue((a[1:]>0).all())
    def test_exact_definition(self):
        """使用直接逐视图其余均值独立核验向量化公式。"""
        rng=np.random.default_rng(9);p=rng.dirichlet(np.ones(10),size=24);c,a=reliability(p)
        for i in range(len(p)):
            block=p[(i//8)*8:(i//8+1)*8];q=np.delete(block,i%8,axis=0).mean(0);m=(p[i]+q)/2
            expected=1-(entropy(m)-(entropy(p[i])+entropy(q))/2)/np.log(2)
            self.assertAlmostEqual(a[i],expected,places=12)
    def test_modes(self):
        """三个单项/联合的权重与指定定义一致。"""
        p=np.random.default_rng(3).dirichlet(np.ones(3),size=16);c,a=reliability(p)
        for mode,r in [('edge-c',c),('edge-v',a),('edge-cv',c*a)]:
            actual=weights(p,mode)[3];self.assertTrue(np.allclose(actual,.5+.5*r));self.assertTrue(((actual>=.5)&(actual<=1)).all())
        self.assertTrue(np.array_equal(weights(p,'baseline')[3],np.ones(16)))
    def test_actual_self_mask(self):
        """自环故意放在第二列，验证不误把首邻居当作自己。"""
        p=np.full((8,2),.5);idx=torch.tensor([[(i+1)%8,i,(i+2)%8] for i in range(8)])
        before=torch.tensor([[1.,.7,.2]]*8,dtype=torch.float16)
        with tempfile.TemporaryDirectory() as d:
            profiler.configure('normal',d);runtime=EdgeRuntime('edge-cv');after=runtime.apply(before,idx,p)
            self.assertTrue(torch.equal(after[:,1],before[:,1]));self.assertTrue(torch.equal(after[:,0],before[:,0]*.5))
            self.assertTrue(torch.equal(before,torch.tensor([[1.,.7,.2]]*8,dtype=torch.float16)))
    def test_disabled_identity(self):
        """Baseline及beta=0直接返回原张量，避免新增舍入。"""
        p=np.full((8,2),.5);idx=torch.arange(8).reshape(8,1);x=torch.ones((8,1),dtype=torch.float16)
        with tempfile.TemporaryDirectory() as d:
            profiler.configure('normal',d)
            self.assertIs(EdgeRuntime('baseline').apply(x,idx,p),x)
            self.assertIs(EdgeRuntime('edge-cv',beta=0).apply(x,idx,p),x)
    def test_invalid_shapes(self):
        """拒绝跨组形状、非法概率及不足两个视图。"""
        for p,v in [(np.full((9,2),.5),8),(np.full((8,2),-.1),8),(np.full((8,2),.5),1)]:
            with self.assertRaises(AssertionError):reliability(p,v)
    def test_control_rng(self):
        """独立打乱按八视图整块进行，保留权重分布及所有全局状态。"""
        p=np.random.default_rng(1).dirichlet(np.ones(3),size=32);state=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
        base=weights(p,'edge-cv')[3];shuffled=weights(p,'shuffled-cv',generator=np.random.default_rng(42))[3]
        self.assertTrue(np.array_equal(np.sort(base),np.sort(shuffled)));self.assertEqual(random.getstate(),state[0]);self.assertTrue(np.array_equal(np.random.get_state()[1],state[1][1]));self.assertTrue(torch.equal(torch.get_rng_state(),state[2]))
        self.assertTrue(np.allclose(weights(p,'uniform')[3],base.mean()))
    def test_algorithm_ast(self):
        """删掉新增节点后，官方距离、对称化、右端、CG与硬化完全一致。"""
        root=Path(__file__).resolve().parents[1];source=(root/'utils/pseudo_labeling.py').read_text()
        tree=InsertWeights().visit(ReplaceCalls().visit(instrument_tree(source)))
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='edge']
        self.assertEqual(sorted(n.func.attr for n in calls),['apply','normalized'])
        restored=Strip().visit(Restore().visit(RemoveObservers().visit(copy.deepcopy(tree))))
        self.assertEqual(ast.dump(restored),ast.dump(ast.parse(source)))
    def test_training_loop(self):
        """新入口的整段适应循环与已验证FAISS入口相同。"""
        root=Path(__file__).resolve().parents[1]
        def loop(name):
            """读取原图批次训练循环。"""
            return ast.dump(next(n for n in ast.parse((root/name).read_text()).body if isinstance(n,ast.For) and 'enumerate(train_loader)' in ast.unparse(n.iter)))
        self.assertEqual(loop('online_weighted_plca.py'),loop('online_faiss.py'))

if __name__=='__main__':unittest.main()
