"""固定缓存图求解器诊断；不导入模型、不适应、不根据真值选择迭代数。"""
import ast
import copy
import datetime
import hashlib
import inspect
import json
from pathlib import Path
import pickle
import sys
import time
import numpy as np
import scipy
import scipy.sparse
from scipy.sparse.linalg import cg,spsolve,eigsh
import torch

LIMITS=[1,2,3,5,10,15,20,30,50,100,200]


def dump(path, obj):
    """保存有限JSON，未知量使用None而不是NaN。"""
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))


def scores_to_labels(scores):
    """沿用官方先转float16再argmax的离散化，不做概率归一化。"""
    return torch.tensor(scores,dtype=torch.float16).argmax(dim=1).numpy()


def metrics(prediction, probabilities, truth):
    """真值只进入离线评价；训练视图标签和原图NMS口径分开。"""
    n,c=probabilities.shape;raw=probabilities.argmax(1);y=np.repeat(truth,8)
    one_hot=np.eye(c)[prediction]
    raw_group=probabilities.reshape(-1,8*c).argmax(1)%c
    corrected_group=(probabilities*one_hot).reshape(-1,8*c).argmax(1)%c
    return dict(views=n,groups=len(truth),view_correct=int((prediction==y).sum()),group_correct=int((corrected_group==truth).sum()),
                view_wrong_to_right=int(((raw!=y)&(prediction==y)).sum()),view_right_to_wrong=int(((raw==y)&(prediction!=y)).sum()),
                group_wrong_to_right=int(((raw_group!=truth)&(corrected_group==truth)).sum()),group_right_to_wrong=int(((raw_group==truth)&(corrected_group!=truth)).sum()))


def stability(label_history, minimum=5, patience=3):
    """只读标签轨迹，不访问真值；未来永久稳定时刻仅作事后观察。"""
    changes=np.count_nonzero(label_history[1:]!=label_history[:-1],axis=1)
    final=label_history[-1];last_different=np.flatnonzero(np.any(label_history!=final,axis=1))
    permanent=int(last_different[-1]+2) if len(last_different) else 1
    candidate=None
    for k in range(max(minimum,patience+1),len(label_history)+1):
        if np.all(changes[k-patience-1:k-1]==0):candidate=k;break
    return dict(changes_between_iterations=changes.tolist(),first_permanent_match_to_final=permanent,
                offline_plateau_candidate=candidate,plateau_differs_from_final=None if candidate is None else int((label_history[candidate-1]!=final).sum()),
                any_later_change_after_plateau=None if candidate is None else bool(np.any(label_history[candidate:]!=label_history[candidate-1])),
                rule='首次k>=5且连续3次所有视图标签不变；不实际终止求解，继续验证未来翻转')


def section(source, first, last):
    """直接提取固定官方版本的顶层语句，避免另写图构建公式。"""
    function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='PLCA')
    return compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy([n for n in function.body if first<=n.lineno<=last]),type_ignores=[])),'official_PLCA_cached_graph','exec')


def build_cache(repo, old, root):
    """从8批缓存重建完全相同的图；没有前向、FAISS或模型更新。"""
    source=(repo/'utils/pseudo_labeling.py').read_text();(root/'official-source.py.txt').write_text(source)
    first=section(source,141,154);second=section(source,156,168)
    cached=old/'smoke/baseline/adaptation';quality=[json.loads(x) for x in (cached/'pseudo_quality.jsonl').read_text().splitlines()]
    normalized=[json.loads(x) for x in (cached/'normalized-graph.jsonl').read_text().splitlines()]
    trace=[json.loads(x) for x in (cached/'input_trace.jsonl').read_text().splitlines()]
    with (repo/'data/cifar-10-batches-py/test_batch').open('rb') as f:truth=np.asarray(pickle.load(f,encoding='latin1')['labels'])
    sampler=torch.utils.data.DistributedSampler(range(10000),num_replicas=1,rank=0,shuffle=True)
    order=np.array(list(sampler));images=np.load(repo/'data/CIFAR-10-C/gaussian_noise.npy',mmap_mode='r')
    directory=root/'fixed-systems';directory.mkdir();checks=[];torch.cuda.set_device(0)
    for step in range(8):
        with np.load(cached/'edge-capture'/f'{step:03d}.npz') as row:
            relation=row['before'];idx=row['indexes'];p=row['p'];assert np.array_equal(row['before'],row['after'])
        indices=order[step*128:(step+1)*128];gold=truth[indices]
        batch=torch.from_numpy(images[40000+indices].transpose(0,3,1,2).copy()).float().div(255)
        h=hashlib.sha256();h.update(batch.numpy().tobytes());h.update(gold.astype(np.int64).tobytes());assert h.hexdigest()==trace[step]['input_sha256'],'缓存标签或样本顺序不匹配'
        namespace=dict(torch=torch,np=np,scipy=scipy,device='cuda',relation=torch.tensor(relation,device='cuda'),Idx=torch.tensor(idx,dtype=torch.long,device='cuda'),X=np.empty((len(p),0),np.float32),K_=idx.shape[1],labels=p)
        exec(first,namespace);S=namespace['Wn'].detach().coalesce()
        assert hashlib.sha256(S.indices().cpu().numpy().tobytes()).hexdigest()==normalized[step]['indices_sha256']
        assert hashlib.sha256(S.values().cpu().numpy().tobytes()).hexdigest()==normalized[step]['values_sha256'],'重建归一化图不一致'
        exec(second,namespace);A=namespace['Af'];labels=namespace['labels'];B=[]
        for c in range(10):
            y=labels[:,c];y=y/(y.sum(dim=-1)+1e-8);B.append(y.cpu().numpy())
        B=np.stack(B,axis=1)
        solutions=np.column_stack([cg(A,B[:,c],tol=1e-6,maxiter=20)[0] for c in range(10)])
        prediction=scores_to_labels(solutions)
        with np.load(cached/'pseudo-inputs'/f'{step:03d}.npz') as row:
            expected=row['pseudo'].argmax(1);assert np.array_equal(prediction,expected),'原版20步标签重放不一致'
        m=metrics(prediction,p,gold)
        assert m['view_correct']==quality[step]['corrected_view_correct'] and m['group_correct']==quality[step]['corrected_group_correct'],'离线标签对应关系不一致'
        np.savez(directory/f'{step:03d}.npz',data=A.data,indices=A.indices,indptr=A.indptr,shape=A.shape,B=B,p=p,truth=gold,original_indices=indices,reference20=prediction)
        checks.append(dict(batch=step,normalized_graph_exact=True,original_inputs_truth_exact=True,reference20_labels_exact=True,A_dtype=str(A.dtype),rhs_dtype=str(B.dtype),solution_dtype=str(solutions.dtype),views=len(p)))
        print('CACHE_VERIFIED '+str(step),flush=True)
    dump(root/'cache-verification.json',checks)


def replay(root):
    """同一A和B从零初值回放；阶段表和参考解不用于精度选参。"""
    out=root/'replay';out.mkdir();all_results=[]
    for file in sorted((root/'fixed-systems').glob('*.npz')):
        with np.load(file) as row:
            A=scipy.sparse.csr_matrix((row['data'],row['indices'],row['indptr']),shape=tuple(row['shape']));B=row['B'];p=row['p'];truth=row['truth'];reference20=row['reference20']
        A64=A.astype(np.float64);B64=B.astype(np.float64);diff=A64-A64.T
        symmetry=float(np.linalg.norm(diff.data)/np.linalg.norm(A64.data));sym=(A64+A64.T)*.5
        eigen=eigsh(sym,k=1,which='SA',v0=np.linspace(1.,2.,sym.shape[0]),return_eigenvectors=False)[0]
        start=time.perf_counter();direct=spsolve(A64,B64);direct_seconds=time.perf_counter()-start
        direct_residual=np.linalg.norm(A64@direct-B64,axis=0)/np.linalg.norm(B64,axis=0);direct_labels=scores_to_labels(direct)
        paths=[];infos=[];callback_seconds=0
        for c in range(10):
            snapshots=[]
            def callback(x):
                """只复制本类迭代值，不读取真值或改变x。"""
                snapshots.append(x.copy())
            start=time.perf_counter();solution,info=cg(A,B[:,c],tol=1e-6,maxiter=200,callback=callback);callback_seconds+=time.perf_counter()-start
            if len(snapshots)==201 and np.array_equal(snapshots[-1],snapshots[-2]):snapshots.pop()
            assert 0<len(snapshots)<=200 and np.array_equal(snapshots[-1],solution)
            infos.append(dict(class_id=c,info=int(info),iterations=len(snapshots),solution_dtype=str(solution.dtype)))
            padded=snapshots+[solution]*(200-len(snapshots));paths.append(np.stack(padded))
        history=np.stack(paths,axis=2)
        labels=np.stack([scores_to_labels(s) for s in history]);assert np.array_equal(labels[19],reference20),'callback迭代编号不匹配20步参考'
        residual=np.linalg.norm(np.stack([A64@F-B64 for F in history]),axis=1)/np.linalg.norm(B64,axis=0)
        phases=[]
        for limit in LIMITS:
            timings=[];plain=None;plain_infos=None
            for timing_repeat in range(3):
                start=time.perf_counter();outputs=[cg(A,B[:,c],tol=1e-6,maxiter=limit) for c in range(10)];timings.append(time.perf_counter()-start)
                plain=np.column_stack([x[0] for x in outputs]);plain_infos=[int(x[1]) for x in outputs]
            assert np.array_equal(plain,history[limit-1]),'callback轨迹与从零重放不一致'
            predicted=labels[limit-1];row=dict(limit=limit,metrics=metrics(predicted,p,truth),different_from_20=int((predicted!=reference20).sum()),different_from_direct=int((predicted!=direct_labels).sum()),relative_residual=residual[limit-1].tolist(),plain_seconds=timings,plain_median_seconds=float(np.median(timings)),info=plain_infos,
                    float64_vs_official_half_argmax_differences=int((plain.argmax(1)!=predicted).sum()))
            phases.append(row)
        permanent=stability(labels);permanent['labels20_different_from_labels200']=int((labels[19]!=labels[-1]).sum())
        if permanent['offline_plateau_candidate'] is not None:
            k=permanent['offline_plateau_candidate'];permanent['candidate_metrics']=metrics(labels[k-1],p,truth)
        result=dict(batch=int(file.stem),views=len(p),groups=len(truth),symmetry_relative_frobenius=symmetry,min_eigenvalue_symmetric_part=float(eigen),A_dtype=str(A.dtype),B_dtype=str(B.dtype),
                    native_200=infos,direct=dict(seconds=direct_seconds,maximum_relative_residual=float(direct_residual.max()),metrics=metrics(direct_labels,p,truth),different_from_20=int((direct_labels!=reference20).sum())),
                    raw=metrics(p.argmax(1),p,truth),phases=phases,stability=permanent,callback_diagnostic_seconds=callback_seconds)
        np.savez(out/f'{file.stem}-labels.npz',labels_by_iteration=labels.astype(np.uint8),relative_residual=residual,direct_labels=direct_labels.astype(np.uint8),reference20=reference20)
        dump(out/f'{file.stem}.json',result);all_results.append(result);print('REPLAY_COMPLETE '+file.stem,flush=True)
    dump(root/'all-graphs.json',all_results)


def main():
    """以时间戳创建新目录，禁止覆盖；不加载或训练神经网络。"""
    home=Path.home();repo=home/'work/IST4TTA-course-project';old=home/'work/ist-edge-reliability-20261117T033732Z'
    root=home/'work'/('ist-cg-replay-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));root.mkdir(exist_ok=False)
    dump(root/'protocol.json',dict(limits=LIMITS,maximum=200,tol=1e-6,original_maxiter=20,graphs=8,raw_images=1024,views=8192,timing_repeats=3,callback='diagnostic only; timing uses native calls without callback',stage_selection='predefined before reading accuracies',plateau_minimum=5,plateau_patience=3,truth='only offline metrics after solving',training=False))
    (root/'cg-installed-source.txt').write_text(inspect.getsource(cg));dump(root/'versions.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__))
    print('RUN_ROOT='+str(root),flush=True);build_cache(repo,old,root);replay(root);dump(root/'complete.json',dict(complete=True,training=False));print('ALL_COMPLETE '+str(root),flush=True)

if __name__=='__main__':main()
