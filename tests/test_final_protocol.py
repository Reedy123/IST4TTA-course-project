"""核查正式模式组合及独立入口的训练、评估逻辑边界。"""
import ast
from pathlib import Path
import unittest
from experiments.final_protocol import MODES, command


class FinalProtocolTests(unittest.TestCase):
    """防止正式整合意外改变已验证逻辑。"""
    def test_modes(self):
        """四模式组成历史读取与资源复用的二乘二消融。"""
        self.assertEqual(len(set(MODES.values())), 4)
        self.assertEqual(MODES['Original'], ('original', 'original'))
        self.assertEqual(MODES['Ours'], ('no-history', 'reuse-resources'))

    def test_command(self):
        """正式运行必须保留完整数据和预训练独立起点。"""
        c = command('/tmp/repo', '/tmp/new', 'Ours', 'fog')
        self.assertEqual(c[c.index('--max_batches')+1], '0')
        self.assertIn('--pretrained', c)
        self.assertNotIn('--model_load_path', c)

    def test_training_unchanged(self):
        """正式入口的适应循环与已验证资源复用入口逐项相同。"""
        root = Path(__file__).resolve().parents[1]
        loops = []
        for name in ['online_faiss.py', 'online_final.py']:
            tree = ast.parse((root/name).read_text())
            loops.append(next(n for n in tree.body if isinstance(n, ast.For)
                              and 'enumerate(train_loader)' in ast.unparse(n.iter)))
        self.assertEqual(ast.dump(loops[0]), ast.dump(loops[1]))

    def test_aggregation_unchanged(self):
        """固定八视图沿用概率平方后最大值规则，禁止换成均值。"""
        text = (Path(__file__).resolve().parents[1]/'experiments/final_evaluate.py').read_text()
        self.assertIn('probabilities.square().flatten(1).argmax(-1) % 10', text)
        self.assertIn('before == after', text)


if __name__ == '__main__':
    unittest.main()
