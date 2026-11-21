"""根据传播前概率计算邻居软权重；不读取真值、不改变全局随机状态。"""
import numpy as np

MODES=('baseline','edge-c','edge-v','edge-cv','uniform','shuffled-cv')


def entropy(p):
    """计算自然对数熵，正确处理零概率。"""
    return -(p*np.log(np.clip(p,1e-12,None))).sum(axis=-1)


def reliability(probabilities, views=8):
    """输入按原图优先排列，返回置信度及留一视图JS一致性。"""
    p=np.asarray(probabilities,dtype=np.float64)
    assert p.ndim==2 and views>1 and len(p)>0 and len(p)%views==0
    assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(-1),1,atol=1e-5)
    groups=p.reshape(-1,views,p.shape[1]);other=(groups.sum(axis=1,keepdims=True)-groups)/(views-1)
    other=np.maximum(other.reshape(p.shape),0)
    midpoint=(p+other)/2
    js=entropy(midpoint)-(entropy(p)+entropy(other))/2
    c=np.clip(1-entropy(p)/np.log(p.shape[1]),0,1)
    a=np.clip(1-js/np.log(2),0,1)
    return c,a


def weights(probabilities, mode='baseline', beta=.5, views=8, generator=None):
    """保留连续权重；Uniform与打乱仅供正向信号后的控制实验。"""
    assert mode in MODES and 0<=beta<1
    c,a=reliability(probabilities,views)
    r={'baseline':np.ones_like(c),'edge-c':c,'edge-v':a}.get(mode,c*a)
    h=1-beta*(1-r)
    if mode=='uniform':h=np.full_like(h,h.mean())
    if mode=='shuffled-cv':
        assert generator is not None,'打乱必须使用独立生成器'
        h=h.reshape(-1,views)[generator.permutation(len(h)//views)].reshape(-1)
    return c,a,r,h


def statistics(values):
    """保存精确分位数和基本分布，避免只看平均值掩盖退化。"""
    v=np.asarray(values,dtype=np.float64).reshape(-1)
    return dict(count=len(v),minimum=float(v.min()),maximum=float(v.max()),mean=float(v.mean()),std=float(v.std()),
                quantiles=dict(zip(['p01','p10','p25','p50','p75','p90','p99'],map(float,np.quantile(v,[.01,.1,.25,.5,.75,.9,.99])))))
