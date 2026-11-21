"""验证插桩可逆、标签等价、嵌套计时及随机状态不变。"""
import ast
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from experiments.profiled_plca import instrument_tree
from experiments.profiling_runtime import Profiler, make_profiled_target
from experiments.target_policies import hard_only

class ProfilingTests(unittest.TestCase):
    """性能观察不能改变算法及随机状态。"""
    def test_reversible(self):
        """还原全部计时作用域后严格等于官方算法AST。"""
        source=(Path(__file__).resolve().parents[1]/'utils/pseudo_labeling.py').read_text()
        tree=instrument_tree(source)
        names=[n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='stage']
        for name in ['faiss','graph','graph_cpu_bridge','propagation_other','feature_forward']:
            self.assertIn(name,names)
    def test_nested_and_rng(self):
        """排他时间求和不重叠，记录过程不消费任何随机数。"""
        with tempfile.TemporaryDirectory() as d, patch('torch.cuda.synchronize'):
            p=Profiler();p.configure('diagnostic',d)
            rng=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
            p.start();p.begin_batch(0,128)
            with p.stage('parent'):
                with p.stage('child'):sum(range(100))
            p.end_batch();p.stop()
            self.assertEqual(rng[0],random.getstate());self.assertTrue(np.array_equal(rng[1][1],np.random.get_state()[1]))
            self.assertTrue(torch.equal(rng[2],torch.get_rng_state()))
            self.assertAlmostEqual(p.records[-1]['inclusive_seconds'],sum(r['exclusive_seconds'] for r in p.records),places=9)
    def test_target(self):
        """Hard-only标签与现有函数完全相同，最终评估不硬化。"""
        class Dataset:
            """最小数据接口替身。"""
            soft_labels=np.array([[.1,.9],[.8,.2]]*8,dtype=np.float32)
            def set_pseudo_labels(self,idx,labels):
                """保留写入结果以验证。"""
                self.labels=labels
        calls=[]
        def robust(*args,**kwargs):
            """记录传播次数，不执行计算。"""
            calls.append(kwargs['repeat']);return Dataset()
        f=make_profiled_target(robust,'hard-only',8)
        r=f(repeat=1);self.assertTrue(np.array_equal(r.labels,hard_only(r.soft_labels,8)))
        f(repeat=0);self.assertEqual(calls,[0,0])

if __name__=='__main__':unittest.main()
