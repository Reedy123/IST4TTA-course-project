"""在官方PLCA内存副本中添加邻居端软权重和只读收敛日志。"""
import ast
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .edge_reliability import weights,statistics
from .faiss_runtime import FaissRuntime,ReplaceCalls
from .profiled_plca import instrument_tree
from .profiling_runtime import profiler,timed_cg


class EdgeRuntime:
    """每次传播前计算权重；不持有模型、不访问真实标签。"""
    def __init__(self, mode, beta=.5, views=8, capture=False):
        """独立生成器只服务控制组，不触碰训练随机数。"""
        self.mode=mode;self.beta=beta;self.views=views;self.capture=capture;self.step=0;self.cg_class=0
        self.generator=np.random.default_rng(420517)

    def write(self, filename, row):
        """在当前全新适应目录追加审计记录。"""
        with (profiler.directory/filename).open('a') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

    def apply(self, relation, indexes, probabilities):
        """三次方距离权重之后仅乘邻居权重，按真实索引保留自环。"""
        start=time.perf_counter();p=np.asarray(probabilities);c,a,r,h=weights(p,self.mode,self.beta,self.views,self.generator)
        assert len(h)==len(relation)
        old=relation.detach().cpu().numpy().copy();idx=indexes.detach().cpu().numpy();self_mask=idx==np.arange(len(h))[:,None]
        if self.mode=='baseline' or self.beta==0:
            result=relation
        else:
            factor=h[idx];factor[self_mask]=1
            result=(relation.float()*torch.as_tensor(factor,dtype=torch.float32,device=relation.device)).to(relation.dtype)
        actual=result.detach().cpu().numpy();assert np.array_equal(actual[self_mask],old[self_mask])
        external=~self_mask;delta=actual.astype(np.float64)-old.astype(np.float64)
        predicted=p.argmax(-1)
        corr=float(np.corrcoef(c,a)[0,1]) if c.std()>0 and a.std()>0 else None
        row=dict(step=self.step,mode=self.mode,beta=self.beta,views=len(h),groups=len(h)//self.views,
                 confidence=statistics(c),agreement=statistics(a),reliability=statistics(r),donor_weight=statistics(h),confidence_agreement_correlation=corr,
                 cv_vs_confidence_weight_mean_difference=float((self.beta*c*(1-a)).mean()),
                 nonself_edges=int(external.sum()),self_edges=int(self_mask.sum()),changed_nonself_edges=int(((actual!=old)&external).sum()),
                 positive_edges_rounded_to_zero=int(((old>0)&(actual==0)&external).sum()),
                 original_nonself_weight=float(old[external].astype(np.float64).sum()),actual_nonself_weight=float(actual[external].astype(np.float64).sum()),
                 absolute_edge_change=float(np.abs(delta[external]).sum()),self_unchanged=True,
                 donor_predicted_class_counts=np.bincount(predicted,minlength=p.shape[1]).tolist(),
                 donor_weight_sum_by_predicted_class=[float(h[predicted==k].sum()) for k in range(p.shape[1])],
                 raw_probability_sha256=hashlib.sha256(p.tobytes()).hexdigest(),edge_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                 calculation_and_readback_seconds=time.perf_counter()-start)
        self.write('edge-reliability.jsonl',row)
        path=profiler.directory/'reliability-arrays';path.mkdir(exist_ok=True)
        np.savez(path/f'{self.step:03d}.npz',confidence=c,agreement=a,reliability=r,donor_weight=h)
        if self.capture:
            path=profiler.directory/'edge-capture';path.mkdir(exist_ok=True)
            np.savez(path/f'{self.step:03d}.npz',before=old,after=actual,indexes=idx,p=p)
        self.step+=1;self.cg_class=0
        return result

    def normalized(self, matrix):
        """只读记录实际归一化稀疏图，coalesce结果不写回传播矩阵。"""
        value=matrix.detach().coalesce();idx=value.indices().cpu().numpy();data=value.values().cpu().numpy()
        assert np.isfinite(data).all() and (data>=0).all()
        diagonal=idx[0]==idx[1]
        self.write('normalized-graph.jsonl',dict(step=self.step-1,stored_entries=len(data),nonzero=int((data!=0).sum()),weights=statistics(data),
                   self_weight=float(data[diagonal].astype(np.float64).sum()),total_weight=float(data.astype(np.float64).sum()),
                   indices_sha256=hashlib.sha256(idx.tobytes()).hexdigest(),values_sha256=hashlib.sha256(data.tobytes()).hexdigest()))

    def cg(self, *args, **kwargs):
        """原CG参数与返回值原样传递；仅离线计算返回解的残差。"""
        start=time.perf_counter();solution,info=timed_cg(*args,**kwargs);elapsed=time.perf_counter()-start
        matrix,rhs=args[:2];norm=float(np.linalg.norm(rhs));residual=float(np.linalg.norm(matrix@solution-rhs))
        self.write('cg-convergence.jsonl',dict(step=self.step-1,category=self.cg_class,info=int(info),relative_residual=residual/(norm+1e-30),
                   rhs_norm=norm,solution_finite=bool(np.isfinite(solution).all()),seconds=elapsed,tolerance=kwargs.get('tol'),maxiter=kwargs.get('maxiter')))
        self.cg_class+=1
        return solution,info


class InsertWeights(ast.NodeTransformer):
    """只在距离三次方之后插入权重，并记录归一化矩阵。"""
    def visit_FunctionDef(self,node):
        """不重写距离、对称化、度、CG右端或硬标签构造。"""
        self.generic_visit(node)
        if node.name=='PLCA':
            for parent in ast.walk(node):
                if not hasattr(parent,'body') or not isinstance(parent.body,list):continue
                body=[]
                for statement in parent.body:
                    body.append(statement)
                    if isinstance(statement,ast.Assign) and statement.lineno==137:
                        body.extend(ast.parse('relation = edge.apply(relation, Idx, dataset.soft_labels)').body)
                    if isinstance(statement,ast.Assign) and statement.lineno==153:
                        body.extend(ast.parse('edge.normalized(Wn)').body)
                parent.body=body
        return node


def load_weighted(path, mode='baseline', beta=.5, views=8, capture=False, audit=False):
    """保留已验证FAISS资源复用，在独立命名空间扩展官方计算。"""
    path=Path(path);tree=InsertWeights().visit(ReplaceCalls().visit(instrument_tree(path.read_text())))
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='stage' and node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value=='faiss':
            for keyword in node.keywords:
                if keyword.arg=='force':keyword.value=ast.Constant(True)
    ast.fix_missing_locations(tree);edge=EdgeRuntime(mode,beta,views,capture)
    namespace={'__name__':'utils.weighted_plca','__package__':'utils','__file__':str(path),'prof':profiler,'timed_cg':edge.cg,'runtime':FaissRuntime('reuse-resources',capture,audit),'edge':edge}
    exec(compile(tree,str(path)+'[weighted]','exec'),namespace)
    return namespace['robust_PLCA']
