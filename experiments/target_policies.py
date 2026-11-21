"""只根据原始概率和PLCA提议选择硬目标，不接收或读取真实标签。"""
import numpy as np


def validate_probabilities(probabilities, views):
    """校验原图优先、视图连续的概率矩阵，支持不满批次。"""
    p=np.asarray(probabilities)
    assert p.ndim==2 and views>0 and len(p)>0 and len(p)%views==0
    assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(-1),1,atol=1e-5)
    return p


def hard_only(probabilities, views=8):
    """将未经图传播的各视图预测独立硬化，不进行跨视图类别重分配。"""
    p=validate_probabilities(probabilities,views)
    return np.eye(p.shape[1],dtype=np.float32)[p.argmax(-1)]


def selective_plca(probabilities, proposals, views=8, confidence=0.8, agreement=0.75):
    """仅接受低原始置信度且获同图多视图支持的冲突修正。"""
    p=validate_probabilities(probabilities,views);q=np.asarray(proposals)
    assert q.shape==p.shape and np.isfinite(q).all()
    assert 0<=confidence<=1 and 0<=agreement<=1
    proposed=q.argmax(-1);raw=p.argmax(-1)
    original_confidence=p.max(-1)
    n,classes=p.shape;groups=n//views
    raw_votes=np.eye(classes,dtype=np.int64)[raw].reshape(groups,views,classes).sum(1)
    # 按原图编号查支持度，绝不把邻近原图的视图混入同组。
    support=raw_votes[np.repeat(np.arange(groups),views),proposed]/views
    conflict=proposed!=raw
    accept=conflict & (original_confidence<confidence) & (support>=agreement)
    selected=np.where(accept,proposed,raw)
    target=np.eye(classes,dtype=np.float32)[selected]
    return target,dict(views=n,groups=groups,plca_proposed_changes=int(conflict.sum()),
                       accepted_changes=int(accept.sum()),rejected_changes=int((conflict & ~accept).sum()),
                       confident_conflict=int((conflict & (original_confidence>=confidence)).sum()),
                       insufficient_support_conflict=int((conflict & (support<agreement)).sum()),
                       accepted_view_indices=np.flatnonzero(accept).tolist(),
                       proposed_view_indices=np.flatnonzero(conflict).tolist())
