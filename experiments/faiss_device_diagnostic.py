"""仅诊断FAISS延迟资源初始化、索引创建及真实CUDA传输，不实现复用。"""
from pathlib import Path
import json
import time
import sys
import numpy as np
import torch
import faiss


def measured(records, name, function):
    """完整设备同步，记录独立调用的墙钟时间。"""
    torch.cuda.synchronize();start=time.perf_counter();result=function();torch.cuda.synchronize()
    records[name]=records.get(name,0)+time.perf_counter()-start
    return result


def main(root):
    """以原版实际产生的8批特征复查，显式初始化只用于归因。"""
    root=Path(root);assert json.loads((root/'diagnostic-original/launcher-timing.json').read_text())['exit_code']==0
    records={};corpus=sorted((root/'diagnostic-original/adaptation/faiss-inputs').glob('*.npz'));assert len(corpus)==8
    for path in corpus:
        with np.load(path) as row:
            X=row['X'];k=int(row['neighbors']);expected=(row['distances'],row['indices'])
        res=measured(records,'resource_python_constructor',faiss.StandardGpuResources)
        measured(records,'explicit_device_resource_initialization',lambda:res.getResources().initializeForDevice(0))
        index=measured(records,'index_constructor_after_device_init',lambda:faiss.GpuIndexFlatL2(res,X.shape[1],faiss.GpuIndexFlatConfig()))
        measured(records,'add_including_H2D',lambda:index.add(X))
        result=measured(records,'search_including_H2D_D2H',lambda:index.search(X,k))
        assert np.array_equal(result[0],expected[0]) and np.array_equal(result[1],expected[1])
        start=time.perf_counter();del index;del res;torch.cuda.synchronize();records['destruction']=records.get('destruction',0)+time.perf_counter()-start
    (root/'device-initialization-diagnostic.json').write_text(json.dumps(dict(batch_count=8,seconds=records,all_search_outputs_exact=True,warning='显式initializeForDevice改变了耗时归属，仅用于诊断；正式对照不预初始化'),ensure_ascii=False,indent=2))
    trace_path=root/'cuda-transfer-trace.json'
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as trace:
        with torch.profiler.record_function('faiss_resource_and_index'):
            res=faiss.StandardGpuResources();index=faiss.GpuIndexFlatL2(res,X.shape[1],faiss.GpuIndexFlatConfig());torch.cuda.synchronize()
        with torch.profiler.record_function('faiss_add'):
            index.add(X);torch.cuda.synchronize()
        with torch.profiler.record_function('faiss_search'):
            result=index.search(X,k);torch.cuda.synchronize()
    trace.export_chrome_trace(str(trace_path))
    events=json.loads(trace_path.read_text())['traceEvents'];copies=[e for e in events if e.get('cat')=='gpu_memcpy'];ranges=[e for e in events if e.get('name') in ['faiss_resource_and_index','faiss_add','faiss_search'] and e.get('ph')=='X'];transfer={}
    for e in copies:
        group=next((r['name'] for r in ranges if r['ts']<=e['ts'] and e['ts']+e.get('dur',0)<=r['ts']+r['dur']),'unassigned')
        key=group+' / '+e['name'];row=transfer.setdefault(key,dict(calls=0,device_milliseconds=0,bytes=0));row['calls']+=1;row['device_milliseconds']+=e.get('dur',0)/1000;row['bytes']+=e.get('args',{}).get('bytes',0)
    summary=dict(gpu_memcpy_events=len(copies),transfer=transfer,note='传输是add/search内的子集，不能与其墙钟再次相加；单批CUPTI诊断不参与性能对照')
    (root/'cuda-transfer-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(dict(initialization=records,copies=summary),ensure_ascii=False,indent=2))

if __name__=='__main__':main(sys.argv[1])
