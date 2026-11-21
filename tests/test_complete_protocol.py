"""核查完整计划、交错顺序和新入口只改变种子及最终保存条件。"""
import ast
from collections import Counter
from pathlib import Path
import unittest
import tempfile
from experiments.complete_protocol import schedule, command


class CompleteTests(unittest.TestCase):
    """避免扩展实验时改变已锁定算法或遗漏某个场景。"""
    def test_schedule(self):
        """180个组合必须唯一，每扰动每种子恰有四模式。"""
        tasks=schedule()
        self.assertEqual(len(tasks),180)
        self.assertTrue(all(n==4 for n in Counter((t['corruption'],t['seed']) for t in tasks).values()))
        self.assertTrue(all(n==45 for n in Counter(t['mode'] for t in tasks).values()))

    def test_interleaving(self):
        """同一场景四模式分散到四轮，避免连续固定模式排序。"""
        tasks=schedule()
        for seed in [42,43,44]:
            a=[t for t in tasks if t['corruption']=='fog' and t['seed']==seed]
            self.assertEqual(sorted(t['round'] for t in a),[0,1,2,3])
            self.assertEqual(len({t['mode'] for t in a}),4)

    def test_command(self):
        """新的扰动和种子可用，其他主要参数沿用原协议。"""
        c=command('/tmp/repo','/tmp/new','Ours','snow',44)
        self.assertEqual(c[c.index('--seed')+1],'44')
        self.assertEqual(c[c.index('--batch_size')+1],'128')
        self.assertIn('--save_model',c)

    def test_algorithm_ast(self):
        """还原保存条件和种子校验后，新旧入口AST完全相同。"""
        root=Path(__file__).resolve().parents[1]
        old=(root/'online_final.py').read_text()
        new=(root/'online_complete.py').read_text()
        self.assertIn('batch_idx + 1 == len(train_loader)',new)
        new=new.replace('assert args.seed in (42, 43, 44) and','assert args.seed == 42 and')
        new=new.replace(' and (batch_idx + 1 == len(train_loader) or (args.max_batches and batch_idx + 1 == args.max_batches))','')
        a=ast.parse(old);b=ast.parse(new)
        a.body=a.body[1:];b.body=b.body[1:]
        self.assertEqual(ast.dump(a),ast.dump(b))

    def test_eval_unchanged(self):
        """特征预处理、固定八视图生成及概率预测函数逐项不变。"""
        root=Path(__file__).resolve().parents[1]
        def definitions(name):
            """只提取评估的纯数据与预测实现，排除命令行元信息。"""
            tree=ast.parse((root/'experiments'/name).read_text())
            return {n.name:ast.dump(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['SharedViews','predict','state_hash']}
        self.assertEqual(definitions('final_evaluate.py'),definitions('complete_evaluate.py'))

    def test_summary(self):
        """三种子样本标准差和配对差异必须按种子聚合，不能混成45次随机重复。"""
        import json
        from experiments.summarize_complete import summarize
        rows=[]
        for task in schedule():
            value=task['seed']-42
            row=dict(**task,single_percent=80+value,eight_percent=85+value,adaptation_seconds=100+value,
                     faiss_seconds=2+value,peak_sampled_gpu_MiB=3000,checkpoint_seconds=.1,
                     legacy_accuracy_reused=False,legacy_timing_reused=False,checkpoint_sha256='test')
            rows.append(row)
        with tempfile.TemporaryDirectory() as d:
            summarize(Path(d),rows,True)
            a=json.loads((Path(d)/'statistics.json').read_text())
            self.assertEqual(a['aggregate']['Ours']['eight_percent'],dict(n=3,mean=86,std=1.0))
            self.assertEqual(a['paired'][-1]['seed_aggregate_difference']['eight_percent']['mean'],0)


if __name__=='__main__':unittest.main()
