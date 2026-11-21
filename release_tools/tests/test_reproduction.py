"""只用CPU的真实Git/命令检查与模拟进程测试；模拟结果绝非GPU冒烟证据。"""
import ast
from copy import deepcopy
import csv
import hashlib
import json
import os
import socket
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

RELEASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(RELEASE/'release_tools'))
from setup_runtime import setup
from locked_source import verify
from prepare_runs import make_plan, execute
from adapt_smoke_entry import smoke_tree
from install_environment import installation_plan
from fetch_weights import prepare_weights
from runtime_support import gpu_idle_info, check_metrics
from summarize_replication import summarize


def metrics(count=128):
    """构造明确的单元测试样本，不读取或改动实验归档。"""
    return dict(count=count,state_unchanged=True,before={'state':'fixture'},after={'state':'fixture'},
                accuracy={'source':{'single':.7,'eight':.8},'ist':{'single':.75,'eight':.85}})


class ReproductionTests(unittest.TestCase):
    """验证可移植调度、源码隔离及失败边界，全程不导入torch、不调用GPU。"""
    @classmethod
    def setUpClass(cls):
        """创建测试专用工作树和空数据目录，验证没有旧16组也能准备计划。"""
        cls.scratch=tempfile.TemporaryDirectory(prefix='ist-cpu-test-')
        cls.root=Path(cls.scratch.name).resolve();cls.runtime=cls.root/'runtime';cls.data=cls.root/'public-data';cls.data.mkdir()
        setup(cls.runtime,cls.data)

    @classmethod
    def tearDownClass(cls):
        """只清理本测试创建的数据链接及临时Git工作树，不接触实际运行目录。"""
        link=cls.runtime/'data'
        if link.is_symlink():link.unlink()
        subprocess.run(['git','-C',str(RELEASE),'worktree','remove',str(cls.runtime)],check=True,capture_output=True)
        cls.scratch.cleanup()

    def setUp(self):
        """为每项测试分配独立输出，不复用已有结果。"""
        self.case=tempfile.TemporaryDirectory(prefix='ist-case-',dir=self.root)
        self.output=Path(self.case.name)/'runs'

    def tearDown(self):
        """清理本项测试的临时文件。"""
        self.case.cleanup()

    def plan(self,**kwargs):
        """以真实锁定源码生成计划，不生成数据或权重。"""
        return make_plan(self.runtime,self.output,sys.executable,
                         ['Original','Original-Reuse','No-history','Ours'],['gaussian_noise'],[42],**kwargs)

    def test_lock_and_idempotent_setup(self):
        """114/92摘要有效，重复setup不覆盖链接。"""
        proof=verify(self.runtime);self.assertEqual((proof['snapshot_files'],proof['source_files']),(114,92))
        self.assertTrue(setup(self.runtime,self.data)['source_unchanged'])

    def test_setup_refuses_other_data(self):
        """已有data指向其他位置时明确停止。"""
        other=Path(self.case.name)/'other-data';other.mkdir()
        with self.assertRaises(ValueError):setup(self.runtime,other)
        self.assertEqual((self.runtime/'data').resolve(),self.data)

    def test_default_four_modes(self):
        """默认四模式使用当前Python，不引用服务器环境路径。"""
        plan=self.plan();self.assertEqual(len(plan['tasks']),4);self.assertEqual(plan['python'],sys.executable)
        self.assertNotIn('/home/featurize',json.dumps(plan));self.assertFalse(self.output.exists())

    def test_original_ours_selection(self):
        """两模式命令支持独立同学从零选择。"""
        p=make_plan(self.runtime,self.output,sys.executable,['Original','Ours'],['fog'],[43])
        self.assertEqual({t['mode'] for t in p['tasks']},{'Original','Ours'})
        self.assertTrue(all(t['seed']==43 for t in p['tasks']))

    def test_full_180_uses_unmodified_entry(self):
        """完整计划维持原参数、原入口、完整评估和180唯一任务。"""
        plan=self.plan(all_180=True);self.assertEqual(len({t['id'] for t in plan['tasks']}),180)
        for t in plan['tasks']:
            cmd=t['adaptation_command'];self.assertIn(str(self.runtime/'online_complete.py'),cmd)
            self.assertEqual(cmd[cmd.index('--max_batches')+1],'0')
            self.assertIn('experiments.complete_evaluate',t['evaluation_command_template'])
        self.assertEqual(plan['evaluation_images'],10000)

    def test_smoke_limits_both_evaluations(self):
        """小测适应2批，初始/最终评估128张，明确替换为外部小测适配器。"""
        plan=self.plan(smoke=True);self.assertEqual(plan['adaptation_batches'],2)
        self.assertEqual(plan['initial_evaluation_images'],128);self.assertEqual(plan['evaluation_images'],128)
        for t in plan['tasks']:
            self.assertIn(str(RELEASE/'release_tools/adapt_smoke_entry.py'),t['adaptation_command'])
            self.assertIn(str(RELEASE/'release_tools/evaluate_subset.py'),t['evaluation_command_template'])

    def test_custom_batches_device_port(self):
        """显式小测规模、设备索引和端口进入计划。"""
        plan=self.plan(max_batches=3,eval_images=256,device=1,master_port=23123)
        self.assertTrue(plan['smoke_only']);self.assertEqual(plan['device'],1)
        cmd=plan['tasks'][0]['adaptation_command'];self.assertEqual(cmd[cmd.index('--master_port')+1],'23123')

    def test_rejects_smoke_180(self):
        """防止小测命令意外铺开180个GPU任务。"""
        with self.assertRaises(ValueError):self.plan(smoke=True,all_180=True)

    def test_invalid_limits(self):
        """拒绝无意义批数、评估规模或未限定的小测。"""
        for args in [dict(max_batches=-1),dict(max_batches=79),dict(eval_images=0),dict(eval_images=128),dict(smoke=True,max_batches=0)]:
            with self.subTest(args=args),self.assertRaises(ValueError):self.plan(**args)

    def test_output_exists_is_untouched(self):
        """拒绝覆盖已有结果目录。"""
        self.output.mkdir();sentinel=self.output/'keep';sentinel.write_text('unchanged')
        with self.assertRaises(ValueError):self.plan()
        self.assertEqual(sentinel.read_text(),'unchanged')

    def test_output_cannot_be_in_data(self):
        """拒绝在数据目录内部写入新实验。"""
        with self.assertRaises(ValueError):make_plan(self.runtime,self.data/'bad',sys.executable,['Ours'],['fog'],[42])

    def test_smoke_ast_only_adds_test_subset(self):
        """去除唯一新增评估限制后，AST与原源码完全一致，文件字节未变。"""
        path=self.runtime/'online_complete.py';raw=path.read_bytes();original=ast.parse(raw)
        changed=smoke_tree(raw,128)
        self.assertEqual(len(changed.body),len(original.body)+1)
        index=next(i for i,n in enumerate(changed.body) if isinstance(n,ast.Assign) and ast.unparse(n).startswith('test_dataset = torch.utils.data.Subset'))
        del changed.body[index];self.assertEqual(ast.dump(changed),ast.dump(original));self.assertEqual(path.read_bytes(),raw)

    def test_smoke_adapter_refuses_wrong_ast(self):
        """原文件结构异常时不猜测改写位置。"""
        with self.assertRaises(ValueError):smoke_tree('x=1',128)

    def test_default_cli_no_execution(self):
        """统一CLI缺省只打印JSON，不探测GPU、不创建运行目录。"""
        result=subprocess.run([sys.executable,str(RELEASE/'reproduce.py'),'run','--repo',str(self.runtime),
                               '--mode','Original','--smoke','--output',str(self.output)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(len(json.loads(result.stdout)['tasks']),1)
        self.assertFalse(self.output.exists())

    def test_busy_gpu_stops(self):
        """所选卡已有进程时不尝试结束进程或启动实验。"""
        with patch('runtime_support.subprocess.check_output',side_effect=['0, UUID0, NVIDIA GeForce RTX 3080, 0, 10240, 0\n','UUID0, 1234\n']):
            with self.assertRaises(ValueError):gpu_idle_info(0)

    def test_scheduler_allocation_is_respected(self):
        """不会通过改CUDA_VISIBLE_DEVICES占用未分配的物理GPU。"""
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'1'}),patch('runtime_support.subprocess.check_output',return_value='0, UUID0, NVIDIA GeForce RTX 3080, 0, 10240, 0\n'):
            with self.assertRaises(ValueError):gpu_idle_info(0)

    def test_busy_port_stops_before_output(self):
        """端口被占用时停止，不结束占用者或创建新实验输出。"""
        with socket.socket() as occupied:
            occupied.bind(('127.0.0.1',0));occupied.listen()
            plan=self.plan(smoke=True,master_port=occupied.getsockname()[1])
            with patch('prepare_runs.check_inputs',return_value={}),patch('prepare_runs.gpu_idle_info',return_value={}),patch('prepare_runs.monitor') as launched:
                with self.assertRaises(OSError):execute(plan)
                launched.assert_not_called();self.assertFalse(self.output.exists())

    def test_failed_environment_retains_failure_not_complete(self):
        """环境检查失败只留失败记录，不标记成功或启动模型。"""
        plan=self.plan(smoke=True,master_port=24124)
        with patch('prepare_runs.check_inputs',return_value={}),patch('prepare_runs.gpu_idle_info',return_value={}),patch('prepare_runs.validate_environment',side_effect=ValueError('CPU UNIT TEST FAILURE')),patch('prepare_runs.monitor') as launched:
            with self.assertRaises(ValueError):execute(plan)
            launched.assert_not_called()
        self.assertTrue((self.output/'failure.json').exists());self.assertFalse((self.output/'complete.json').exists())

    def test_other_idle_gpu_is_not_paper_benchmark(self):
        """可在其他空闲NVIDIA卡做流程验证，但不标作3080论文硬件。"""
        with patch('runtime_support.subprocess.check_output',side_effect=['1, UUID1, NVIDIA GeForce RTX 3060, 0, 12288, 0\n','']):
            result=gpu_idle_info(1);self.assertEqual(result['index'],1);self.assertFalse(result['paper_hardware'])

    def test_bad_metrics_rejected(self):
        """样本数、状态或数值无效就不生成成功结果。"""
        a=metrics();a['count']=10;b=metrics();b['after']={'state':'changed'};c=metrics();c['accuracy']['ist']['eight']=float('nan')
        for result in [a,b,c]:
            with self.subTest(result=result),self.assertRaises(ValueError):check_metrics(result,128)

    def test_installer_refuses_existing_env(self):
        """安装入口不会覆盖已有环境。"""
        with self.assertRaises(ValueError):installation_plan(Path(self.case.name))

    def test_installer_plan_is_portable(self):
        """安装命令为conda显式包加pip来源锁，前缀由用户提供。"""
        plan=installation_plan(self.output);self.assertEqual(plan[0][:3],['conda','create','-y'])
        self.assertIn(str(self.output),plan[0]);self.assertFalse(self.output.exists())

    def test_busy_execute_creates_nothing(self):
        """执行前GPU检查失败时不创建输出，也不运行适应进程。"""
        plan=self.plan(smoke=True)
        with patch('prepare_runs.check_inputs',return_value={}),patch('prepare_runs.gpu_idle_info',side_effect=ValueError('busy')),patch('prepare_runs.monitor') as launched:
            with self.assertRaises(ValueError):execute(plan)
            launched.assert_not_called();self.assertFalse(self.output.exists())

    def test_cpu_mock_orchestration_and_summary(self):
        """模拟子进程输出验证调度/评估/汇总衔接；不表示GPU或模型通过。"""
        plan=make_plan(self.runtime,self.output,sys.executable,['Ours'],['gaussian_noise'],[42],smoke=True,master_port=24123)
        hardware={'name':'CPU-UNIT-TEST-FIXTURE','paper_hardware':False}
        def fake_adaptation(cmd,log,samples,repo,device):
            """只在临时目录生成明确的模拟文件。"""
            target=Path(cmd[cmd.index('--save_path')+1]);target.mkdir()
            (target/'fixture_adapt.pth').write_bytes(b'CPU TEST ONLY - NOT A MODEL')
            (target/'profile-batches.jsonl').write_text('{"images":128}\n'*2)
            (target/'profile-summary.json').write_text(json.dumps({'adaptation_wall_seconds':1.0,'stage_inclusive_seconds':{'faiss':.1},'checkpoint_block_seconds':.01}))
            return {'exit_code':0,'peak_sampled_gpu_MiB':None}
        def fake_evaluation(cmd,*args,**kwargs):
            """只模拟有效评估记录，不调用torch或NVIDIA。"""
            target=Path(cmd[cmd.index('--out')+1]);target.mkdir();(target/'results.json').write_text(json.dumps(metrics()))
            (target/'predictions.csv').write_text('index,label\n'+''.join(f'{i},0\n' for i in range(128)))
            return subprocess.CompletedProcess(cmd,0)
        with patch('prepare_runs.check_inputs',return_value={}),patch('prepare_runs.gpu_idle_info',return_value=hardware),patch('prepare_runs.validate_environment',return_value={'CPU_mock_only':True}),patch('prepare_runs.monitor',side_effect=fake_adaptation),patch('prepare_runs.evaluate_command',side_effect=fake_evaluation):
            execute(plan)
        result=summarize(self.output,Path(self.case.name)/'summary');self.assertTrue(result['smoke_only'])
        self.assertIsNone(result['summary'][0]['eight_percent_sample_sd']);self.assertFalse(result['paper_hardware'])
        with self.assertRaises(ValueError):summarize(self.output,Path(self.case.name)/'summary')
        with self.assertRaises(ValueError):self.plan()

    def test_public_weight_preparation_no_model_load(self):
        """真实Git公开权重可提取和重复核验，错误文件不会被覆盖；不加载模型。"""
        data=Path(self.case.name)/'weight-data';data.mkdir()
        record=prepare_weights(data);self.assertEqual(record['bytes'],9039237)
        self.assertTrue(prepare_weights(data)['already_present'])
        target=data/'ModelZoo/Hendrycks2020AugMixWRN.pt';target.write_bytes(b'unit-test-invalid')
        with self.assertRaises(ValueError):prepare_weights(data)
        self.assertEqual(target.read_bytes(),b'unit-test-invalid')

    def test_no_torch_imported(self):
        """本测试集仅CPU标准库路径，不以模拟结果冒称GPU验证。"""
        self.assertNotIn('torch',sys.modules);self.assertNotIn('faiss',sys.modules)


if __name__=='__main__':
    unittest.main()
