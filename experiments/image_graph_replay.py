"""缓存特征上的原图级PLCA回放；不加载模型、不训练、不按准确率选择K。"""
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import types
import numpy as np
import torch
from scipy.sparse.linalg import cg
from .faiss_runtime import load_faiss_profiled
from .cg_replay import metrics,dump


MODES=('view-plca','image-k6','image-k50')


def fuse(features, probabilities, views=8):
    """原图优先、同图视图连续；仅float32算术平均，无归一化或筛选。"""
    x=np.asarray(features);p=np.asarray(probabilities)
    assert x.ndim==p.ndim==2 and len(x)==len(p) and len(x)%views==0
    return x.reshape(-1,views,x.shape[1]).mean(axis=1,dtype=np.float32),p.reshape(-1,views,p.shape[1]).mean(axis=1,dtype=np.float32)


def broadcast_labels(labels, views=8):
    """原图one-hot类别按原索引映射回全部增强视图。"""
    return np.repeat(np.asarray(labels),views)


def pool_view_scores(scores, views=8):
    """对官方半精度传播分数取float32均值，再共享类别，用于隔离标签共享效应。"""
    s=np.asarray(scores);return s.reshape(-1,views,s.shape[1]).mean(axis=1,dtype=np.float32).argmax(1)


def neighbor_statistics(indexes, relation, views=8):
    """统计非自环同图邻居、正边质量和不同原图覆盖，不访问真实标签。"""
    idx=np.asarray(indexes);w=np.asarray(relation,dtype=np.float64);query=np.arange(len(idx))[:,None]
    self_mask=idx==query;same=idx//views==query//views;other_same=same&~self_mask;nonself=~self_mask
    unique=[len(np.unique(row[row//views!=i//views]//views)) for i,row in enumerate(idx)]
    def summarize(a):
        """保留分位数，避免只用总比例掩盖样本差异。"""
        return dict(mean=float(np.mean(a)),minimum=float(np.min(a)),median=float(np.median(a)),maximum=float(np.max(a)))
    positive=(w>0)&nonself
    return dict(nodes=len(idx),retrieved=idx.shape[1],self_count=int(self_mask.sum()),nonself_slots=int(nonself.sum()),other_same_image_slots=int(other_same.sum()),
                other_same_image_slot_fraction=float(other_same.sum()/nonself.sum()),same_image_positive_edges=int((positive&same).sum()),positive_nonself_edges=int(positive.sum()),
                same_image_positive_edge_fraction=float((positive&same).sum()/positive.sum()),same_image_nonself_weight=float(w[other_same].sum()),all_nonself_weight=float(w[nonself].sum()),same_image_weight_fraction=float(w[other_same].sum()/w[nonself].sum()),
                other_same_neighbors_per_view=summarize(other_same.sum(1)),distinct_other_images_per_view=summarize(unique),
                top8_same_image_slots=int(other_same[:,:8].sum()),top8_nonself_slots=int(nonself[:,:8].sum()),same_image_fraction_top8_slots_excluding_self=float(other_same[:,:8].sum()/nonself[:,:8].sum()))


class CachedDataset:
    """仅提供官方PLCA所需的特征与概率接口，不含真值。"""
    def __init__(self, x, p):
        """每次新建标签容器，防止上次one-hot写回影响下一次回放。"""
        self.features=list(x);self.pseudo_labels=list(p)
    def set_pseudo_labels(self, indexes, labels):
        """原样接收官方硬标签，按节点索引写回。"""
        for i,label in zip(indexes,labels):self.pseudo_labels[i]=label


class SolverCapture:
    """只收集原始CG返回分数及计时，原参数不变，不研究或调整求解器。"""
    def __init__(self):
        """建立当前回放的临时记录。"""
        self.reset()
    def reset(self):
        """不同图之间清空，避免混入旧结果。"""
        self.solutions=[];self.infos=[];self.seconds=0.;self.last_search=None
    def __call__(self,*args,**kwargs):
        """保留tol=1e-6,maxiter=20，未经改动地调用原SciPy。"""
        start=time.perf_counter();f,info=cg(*args,**kwargs);self.seconds+=time.perf_counter()-start
        self.solutions.append(f);self.infos.append(int(info));return f,info


def run_pipeline(plca, capture, x, p, mode):
    """同步整个图流程计墙钟，包含融合、FAISS、构图、CG及标签映射。"""
    torch.cuda.synchronize();start=time.perf_counter();capture.reset()
    if mode=='view-plca':features,probabilities=x,p;K=50
    else:features,probabilities=fuse(x,p);K=6 if mode=='image-k6' else 50
    assert K+2<=len(features)
    dataset=CachedDataset(features,probabilities)
    plca(dataset,bank=([],[]),K=K,gamma=3,mode='l2',repeat=1,device='cuda',return_dst=True,verbose=False)
    labels=np.array(dataset.pseudo_labels).argmax(1)
    if mode!='view-plca':labels=broadcast_labels(labels)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    return dict(labels=labels,seconds=elapsed,cg_seconds=capture.seconds,cg_info=list(capture.infos),nodes=len(features),K=K,retrieved=K+2)


def main():
    """固定两种K规则和全部对照，缓存回放后仅输出继续/停止判断。"""
    home=Path.home();repo=home/'work/IST4TTA-course-project';old=home/'work/ist-edge-reliability-20261117T033732Z/smoke/baseline/adaptation';systems=home/'work/ist-cg-replay-20261117T041824Z/fixed-systems'
    root=home/'work'/('ist-image-plca-replay-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));root.mkdir(exist_ok=False)
    config=dict(primary='image-k6',density_rule='round(50/8)=6, preserves approximate positive nonself degree fraction',fixed_count_control='image-k50',feature_fusion='float32 arithmetic mean; no L2 normalization',probability_fusion='float32 arithmetic mean of original per-view probabilities',mapping='one-hot argmax image label repeated to original 8 views; original per-view KL targets unchanged in any future adaptation',
                controls=['fusion-only','view-plca-score-pool'],repeat=1,cg_maxiter=20,cg_tol=1e-6,gamma=3,alpha=.99,faiss='exact FlatL2, resource reuse, fresh empty index every call',batches=8,raw_images=1024,views=8192,timing_repeats=6,warmups_per_case_per_graph=1,
                continuation_rule='primary K6 group_correct>=view baseline, view_correct>=view baseline, group_correct>fusion-only, graph pipeline median sum <=80% view baseline; engineering signal only, no significance claim',training=False,parameter_search=False)
    dump(root/'protocol.json',config);print('RUN_ROOT='+str(root),flush=True)
    robust=load_faiss_profiled(repo/'utils/pseudo_labeling.py','reuse-resources',False,False);namespace=robust.__globals__;plca=namespace['PLCA']
    namespace['dist']=types.SimpleNamespace(get_rank=lambda:0)
    capture=SolverCapture();namespace['timed_cg']=capture;torch.cuda.set_device(0)
    original_search=namespace['runtime'].search
    def audit_search(index,vectors,neighbors):
        """只保留原生查询输出用于与缓存逐元素核验。"""
        result=original_search(index,vectors,neighbors);capture.last_search=result;return result
    namespace['runtime'].search=audit_search
    audit=[json.loads(s) for s in (old/'faiss-audit.jsonl').read_text().splitlines()];results=[]
    for step in range(8):
        with np.load(old/'faiss-inputs'/f'{step+1:03d}.npz') as row:
            x=row['X'];ref_idx=row['indices'];ref_distance=row['distances']
        with np.load(old/'edge-capture'/f'{step:03d}.npz') as row:
            p=row['p'];idx=row['indexes'];relation=row['before'];assert np.array_equal(idx,ref_idx)
        assert hashlib.sha256(x.tobytes()).hexdigest()==audit[step]['feature_sha256']
        with np.load(systems/f'{step:03d}.npz') as row:truth=row['truth'];reference20=row['reference20']
        stats=neighbor_statistics(idx,relation);warm={};timings={m:[] for m in MODES};outputs={}
        for mode in MODES:
            warm[mode]=run_pipeline(plca,capture,x,p,mode)
            if mode=='view-plca':
                assert np.array_equal(capture.last_search[0],ref_distance) and np.array_equal(capture.last_search[1],ref_idx),'原视图FAISS结果未精确复现'
        assert np.array_equal(warm['view-plca']['labels'],reference20),'视图图基线标签与缓存不同'
        for repetition in range(6):
            order=MODES[repetition%3:]+MODES[:repetition%3]
            for mode in order:
                output=run_pipeline(plca,capture,x,p,mode);assert np.array_equal(output['labels'],warm[mode]['labels'])
                timings[mode].append(dict(seconds=output['seconds'],cg_seconds=output['cg_seconds']));outputs[mode]=output
        start=time.perf_counter();pf=p.reshape(-1,8,p.shape[1]).mean(axis=1,dtype=np.float32);fusion_labels=broadcast_labels(pf.argmax(1));fusion_seconds=time.perf_counter()-start
        capture.reset();run_pipeline(plca,capture,x,p,'view-plca')
        scores=torch.tensor(np.column_stack(capture.solutions),dtype=torch.float16).numpy();assert np.array_equal(scores.argmax(1),reference20)
        start=time.perf_counter();pooled_labels=broadcast_labels(pool_view_scores(scores));pool_seconds=time.perf_counter()-start
        groups={}
        for mode in MODES:
            out=outputs[mode];groups[mode]=dict(metrics=metrics(out['labels'],p,truth),timings=timings[mode],median_seconds=float(np.median([s['seconds'] for s in timings[mode]])),median_cg_seconds=float(np.median([s['cg_seconds'] for s in timings[mode]])),nodes=out['nodes'],K=out['K'],retrieved=out['retrieved'],cg_info=out['cg_info'])
        groups['fusion-only']=dict(metrics=metrics(fusion_labels,p,truth),median_seconds=fusion_seconds,nodes=0,K=None)
        groups['view-plca-score-pool']=dict(metrics=metrics(pooled_labels,p,truth),median_seconds=groups['view-plca']['median_seconds']+pool_seconds,nodes=1024,K=50)
        np.savez(root/f'{step:03d}-labels.npz',truth=truth,p=p,reference=reference20,**{mode.replace('-','_'):out['labels'] for mode,out in outputs.items()},fusion_only=fusion_labels,view_score_pool=pooled_labels)
        record=dict(batch=step,feature_sha256=audit[step]['feature_sha256'],reference_labels_exact=True,reference_faiss_exact=True,neighbors=stats,warmup_seconds={m:v['seconds'] for m,v in warm.items()},groups=groups)
        results.append(record);dump(root/f'{step:03d}.json',record);print('REPLAY_COMPLETE '+str(step),flush=True)
    summary={}
    for mode in list(MODES)+['fusion-only','view-plca-score-pool']:
        rows=[r['groups'][mode] for r in results];keys=rows[0]['metrics'].keys();summary[mode]=dict(metrics={k:sum(r['metrics'][k] for r in rows) for k in keys},graph_pipeline_seconds=sum(r['median_seconds'] for r in rows))
    primary=summary['image-k6'];base=summary['view-plca'];fusion=summary['fusion-only']
    conditions=dict(group_not_below_view=primary['metrics']['group_correct']>=base['metrics']['group_correct'],view_not_below_view=primary['metrics']['view_correct']>=base['metrics']['view_correct'],group_above_fusion=primary['metrics']['group_correct']>fusion['metrics']['group_correct'],warm_graph_saving_at_least20percent=primary['graph_pipeline_seconds']<=.8*base['graph_pipeline_seconds'])
    summary_result=dict(groups=summary,conditions=conditions,reasonable_tradeoff=all(conditions.values()),note='仅针对事先固定K6的缓存诊断，不按准确率在K6/K50中挑选；没有运行完整适应')
    dump(root/'all-graphs.json',results);dump(root/'summary.json',summary_result);dump(root/'complete.json',dict(complete=True,training=False));print('ALL_COMPLETE '+str(root)+' '+json.dumps(summary_result),flush=True)

if __name__=='__main__':main()
