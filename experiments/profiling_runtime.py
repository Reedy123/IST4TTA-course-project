"""分离同步诊断与正常墙钟计时，不改变任何训练计算或随机状态。"""
from contextlib import contextmanager
from pathlib import Path
import json
import time
import numpy as np
import torch
from scipy.sparse.linalg import cg as native_cg
from .target_policies import hard_only


class Profiler:
    """统计嵌套阶段的排他时间，避免重复计算子阶段。"""
    def __init__(self):
        """创建不访问GPU、不消耗随机数的记录器。"""
        self.active=False;self.kind='normal';self.directory=None;self.stack=[];self.records=[];self.io_seconds=0.

    def configure(self, kind, directory):
        """在训练开始前固定模式和全新结果路径。"""
        assert kind in ['diagnostic','normal']
        self.kind=kind;self.directory=Path(directory)

    def start(self):
        """完整适应起点；初始评估不计入适应时间。"""
        torch.cuda.synchronize();self.active=True;self.started=time.perf_counter()

    @contextmanager
    def stage(self, name, force=False, gpu=True):
        """诊断同步全设备流；正常模式只单列原检查点保存块。"""
        if not self.active or (self.kind=='normal' and not force):
            yield
            return
        if self.kind=='diagnostic' and gpu:torch.cuda.synchronize()
        start=time.perf_counter();frame=[name,0.];self.stack.append(frame)
        try:
            yield
        finally:
            if self.kind=='diagnostic' and gpu:torch.cuda.synchronize()
            elapsed=time.perf_counter()-start;self.stack.pop()
            if self.stack:self.stack[-1][1]+=elapsed
            self.records.append(dict(batch=self.batch,name=name,inclusive_seconds=elapsed,exclusive_seconds=elapsed-frame[1]))
            if name=='checkpoint_io':self.io_seconds+=elapsed

    def begin_batch(self, step, images):
        """保存批大小，用于区分最后一个小批次。"""
        self.batch=step;self.images=images;self.batch_started=time.perf_counter()

    def end_batch(self):
        """记录批次主体时间；外层取数等待仍归入完整区间其他开销。"""
        row=dict(batch=self.batch,images=self.images,body_seconds=time.perf_counter()-self.batch_started)
        with (self.directory/'profile-batches.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')

    def stop(self):
        """立即写盘计时；完整脚本后续评估不计入统计。"""
        torch.cuda.synchronize();elapsed=time.perf_counter()-self.started;self.active=False
        totals={};counts={};inclusive={}
        for r in self.records:
            totals[r['name']]=totals.get(r['name'],0)+r['exclusive_seconds']
            inclusive[r['name']]=inclusive.get(r['name'],0)+r['inclusive_seconds']
            counts[r['name']]=counts.get(r['name'],0)+1
        attributed=sum(totals.values());other=elapsed-attributed
        assert other>=-1e-5
        summary=dict(kind=self.kind,adaptation_wall_seconds=elapsed,stage_exclusive_seconds=totals,
                     stage_inclusive_seconds=inclusive,stage_calls=counts,unattributed_other_seconds=other,
                     checkpoint_block_seconds=self.io_seconds,normal_stages_interpretable=self.kind=='diagnostic',
                     note='诊断GPU阶段边界全设备同步；SciPy仅计CPU调用。正常仅起止同步，保存块单列但含可能的GPU等待。')
        (self.directory/'profile-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
        (self.directory/'profile-stages.json').write_text(json.dumps(self.records,indent=2))


profiler=Profiler()


def timed_cg(*args, **kwargs):
    """仅围住CPU SciPy调用，参数及返回完全沿用官方实现。"""
    with profiler.stage('scipy_cg',gpu=False):
        return native_cg(*args,**kwargs)


def make_profiled_target(robust, flow, views):
    """复用同一预测，Hard-only仅跳过传播并生成原始硬标签。"""
    assert flow in ['no-history','hard-only']
    def target(*args,**kwargs):
        """最终repeat=0路径不套用硬化，适应定义与既有对照相同。"""
        adaptation=kwargs.get('repeat',1)>0
        options=dict(kwargs)
        if adaptation and flow=='hard-only':options['repeat']=0
        result=robust(*args,**options)
        if adaptation and flow=='hard-only':
            with profiler.stage('target_hardening',gpu=False):
                p=np.asarray(result.soft_labels);labels=hard_only(p,views)
                result.set_pseudo_labels(list(range(len(labels))),labels)
        return result
    return target
