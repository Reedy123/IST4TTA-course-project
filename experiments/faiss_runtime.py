"""诊断并按需复用FAISS资源；精确Flat搜索及官方图计算保持原样。"""
import ast
import hashlib
import json
import threading
from pathlib import Path
import numpy as np
import faiss
from .profiled_plca import instrument_tree
from .profiling_runtime import profiler, timed_cg


class FaissRuntime:
    """单进程、单调用线程资源容器；索引始终每批新建。"""
    def __init__(self, mode='original', capture=False, audit=False):
        """只复用资源对象，不缓存索引或批次特征。"""
        assert mode in ['original','reuse-resources']
        self.mode=mode;self.capture=capture;self.audit=audit;self.cached=None;self.step=0;self.owner=None

    def resources(self):
        """资源构造本身可能不分配GPU；延迟初始化记在索引创建中。"""
        with profiler.stage('faiss_resource_constructor'):
            if self.mode=='original':return faiss.StandardGpuResources()
            caller=threading.get_ident()
            if self.cached is None:
                self.cached=faiss.StandardGpuResources();self.owner=caller
            assert self.owner==caller, '同一FAISS资源不得跨CPU线程并发复用'
            return self.cached

    def index(self, resource, dimensions, config, metric):
        """保持默认精度、设备、距离类型；每次取得全新空索引。"""
        with profiler.stage('faiss_index_constructor_lazy_init'):
            cls=faiss.GpuIndexFlatL2 if metric=='l2' else faiss.GpuIndexFlatIP
            result=cls(resource,dimensions,config)
            assert result.ntotal==0,'新批次索引必须为空'
            return result

    def add(self, index, vectors):
        """索引无旧数据；add包含官方CPU特征到GPU索引存储的复制。"""
        assert index.ntotal==0
        with profiler.stage('faiss_add_including_transfer'):
            index.add(vectors)
        assert index.ntotal==len(vectors)

    def search(self, index, vectors, neighbors):
        """保留原始查询，返回原始NumPy距离与索引，不作排序修正。"""
        assert index.ntotal==len(vectors)
        with profiler.stage('faiss_search_including_transfer'):
            result=index.search(vectors,neighbors)
        if profiler.active:
            self.step+=1
            if self.audit:
                row=dict(step=self.step,count=len(vectors),index_count=int(index.ntotal),neighbors=int(neighbors),feature_sha256=hashlib.sha256(vectors.tobytes()).hexdigest(),distance_sha256=hashlib.sha256(result[0].tobytes()).hexdigest(),index_sha256=hashlib.sha256(result[1].tobytes()).hexdigest())
                with (profiler.directory/'faiss-audit.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if self.capture:
                with profiler.stage('faiss_capture_io',gpu=False):
                    path=profiler.directory/'faiss-inputs';path.mkdir(exist_ok=True)
                    np.savez(path/f'{self.step:03d}.npz',X=vectors,distances=result[0],indices=result[1],neighbors=neighbors)
        return result


class ReplaceCalls(ast.NodeTransformer):
    """仅把FAISS调用委托给透明包装；其余官方语句不改。"""
    def visit_Call(self,node):
        """调用顺序、传参及返回值保持原样。"""
        self.generic_visit(node);f=node.func
        if isinstance(f,ast.Attribute) and isinstance(f.value,ast.Name):
            if f.value.id=='faiss' and f.attr=='StandardGpuResources':
                node.func=ast.Attribute(value=ast.Name(id='runtime',ctx=ast.Load()),attr='resources',ctx=ast.Load())
            elif f.value.id=='faiss' and f.attr in ['GpuIndexFlatL2','GpuIndexFlatIP']:
                node.func=ast.Attribute(value=ast.Name(id='runtime',ctx=ast.Load()),attr='index',ctx=ast.Load())
                node.args.append(ast.Constant('l2' if f.attr=='GpuIndexFlatL2' else 'ip'))
            elif f.value.id=='index' and f.attr in ['add','search']:
                node.func=ast.Attribute(value=ast.Name(id='runtime',ctx=ast.Load()),attr=f.attr,ctx=ast.Load());node.args.insert(0,ast.Name(id='index',ctx=ast.Load()))
        return node


def load_faiss_profiled(path, mode='original', capture=False, audit=False):
    """载入官方计算的内存副本，FAISS整块在正常模式也计墙钟。"""
    path=Path(path);tree=ReplaceCalls().visit(instrument_tree(path.read_text()))
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='stage' and node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value=='faiss':
            for keyword in node.keywords:
                if keyword.arg=='force':keyword.value=ast.Constant(True)
    ast.fix_missing_locations(tree)
    namespace={'__name__':'utils.faiss_profiled','__package__':'utils','__file__':str(path),'prof':profiler,'timed_cg':timed_cg,'runtime':FaissRuntime(mode,capture,audit)}
    exec(compile(tree,str(path)+'[faiss-profiled]','exec'),namespace)
    return namespace['robust_PLCA']
