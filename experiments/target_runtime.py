"""在独立入口复用官方特征提取和图传播，仅替换训练伪标签。"""
import json
from pathlib import Path
import numpy as np
from utils.pseudo_labeling import robust_PLCA
from .target_policies import hard_only, selective_plca


def make_target_policy(mode, views, log=None, confidence=0.8, agreement=0.75):
    """建立无随机操作的目标策略；最终评估repeat=0仍调用原路径。"""
    assert mode in ['hard-only','selective-plca']
    step=0
    def policy(*args, **kwargs):
        """PLCA前概率按已恢复索引取用，不访问数据集真实标签。"""
        nonlocal step
        if kwargs.get('repeat',1)<=0:
            return robust_PLCA(*args,**kwargs)
        options=dict(kwargs)
        if mode=='hard-only':options['repeat']=0
        result=robust_PLCA(*args,**options)
        probabilities=np.asarray(result.soft_labels)
        if mode=='hard-only':
            assert np.array_equal(probabilities,np.asarray(result.pseudo_labels))
            target=hard_only(probabilities,views)
            record=dict(views=len(target),groups=len(target)//views,plca_proposed_changes=0,
                        accepted_changes=0,rejected_changes=0,graph_executed=False)
        else:
            target,record=selective_plca(probabilities,result.pseudo_labels,views,confidence,agreement)
            record['graph_executed']=True
        assert target.shape==probabilities.shape and target.dtype==np.float32
        assert np.all(target.sum(-1)==1) and np.all((target==0)|(target==1))
        result.set_pseudo_labels(list(range(len(target))),target)
        record.update(step=step,mode=mode,confidence_threshold=confidence,agreement_threshold=agreement,
                      target_kind='per-view one-hot',diagnostic_uses_true_labels=False)
        if log:
            with Path(log).open('a') as file:file.write(json.dumps(record)+'\n')
        step+=1
        return result
    return policy
