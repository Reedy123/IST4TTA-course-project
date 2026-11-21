"""验证资源生命周期、索引隔离及算法替换边界。"""
import ast
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from experiments.faiss_runtime import FaissRuntime, ReplaceCalls
from experiments.profiled_plca import instrument_tree, Strip

class Restore(ast.NodeTransformer):
    """移除调用委托，验证原始算法结构。"""
    def visit_Call(self,node):
        """将包装调用还原为原FAISS调用。"""
        self.generic_visit(node);f=node.func
        if isinstance(f,ast.Attribute) and isinstance(f.value,ast.Name) and f.value.id=='runtime':
            if f.attr=='resources':node.func=ast.Attribute(value=ast.Name(id='faiss',ctx=ast.Load()),attr='StandardGpuResources',ctx=ast.Load())
            elif f.attr=='index':
                metric=node.args.pop().value;node.func=ast.Attribute(value=ast.Name(id='faiss',ctx=ast.Load()),attr='GpuIndexFlatL2' if metric=='l2' else 'GpuIndexFlatIP',ctx=ast.Load())
            else:
                node.args.pop(0);node.func=ast.Attribute(value=ast.Name(id='index',ctx=ast.Load()),attr=f.attr,ctx=ast.Load())
        return node

class FaissTests(unittest.TestCase):
    """只复用资源，禁止跨批次复用数据。"""
    def test_resource_lifetime(self):
        """原版每批新建，优化版相同线程只创建一次。"""
        with patch('experiments.faiss_runtime.faiss.StandardGpuResources',side_effect=lambda:object()) as ctor:
            old=FaissRuntime();self.assertIsNot(old.resources(),old.resources())
            new=FaissRuntime('reuse-resources');self.assertIs(new.resources(),new.resources());self.assertEqual(ctor.call_count,3)
            with patch('experiments.faiss_runtime.threading.get_ident',return_value=-1):
                with self.assertRaises(AssertionError):new.resources()
    def test_fresh_index(self):
        """每批新索引，允许最后一个小批次，拒绝脏索引。"""
        class Index:
            """仅模拟索引计数，不计算近邻。"""
            ntotal=0
            def add(self,x):
                """模拟FAISS插入计数。"""
                self.ntotal+=len(x)
        with patch('experiments.faiss_runtime.faiss.GpuIndexFlatL2',side_effect=lambda *args:Index()):
            r=FaissRuntime('reuse-resources');a=r.index(None,128,None,'l2');b=r.index(None,128,None,'l2')
            self.assertIsNot(a,b);r.add(a,np.zeros((1024,128),np.float32));r.add(b,np.zeros((128,128),np.float32))
            self.assertEqual((a.ntotal,b.ntotal),(1024,128))
            with self.assertRaises(AssertionError):r.add(a,np.zeros((1,128),np.float32))
    def test_exact_ast(self):
        """恢复包装后，官方全部函数AST完全相同。"""
        source=(Path(__file__).resolve().parents[1]/'utils/pseudo_labeling.py').read_text()
        tree=ReplaceCalls().visit(instrument_tree(source));tree=Strip().visit(Restore().visit(copy.deepcopy(tree)))
        self.assertEqual(ast.dump(ast.parse(source)),ast.dump(tree))
    def test_training_loop(self):
        """完整运行的训练、损失与PMA主体不变；小测截断单独剔除。"""
        root=Path(__file__).resolve().parents[1]
        def loop(name):
            """提取原图批次循环，去掉仅小测使用的退出条件。"""
            tree=ast.parse((root/name).read_text());node=next(n for n in tree.body if isinstance(n,ast.For) and 'enumerate(train_loader)' in ast.unparse(n.iter))
            node.body=[n for n in node.body if not (isinstance(n,ast.If) and 'args.max_batches' in ast.unparse(n.test))]
            return ast.dump(node)
        self.assertEqual(loop('online_profile.py'),loop('online_faiss.py'))

if __name__=='__main__':unittest.main()
