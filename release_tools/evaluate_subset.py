"""小测专用固定子集评估：复用锁定评估函数，不改原评估文件或论文统计。"""
import argparse
import csv
import json
import random
from pathlib import Path
import sys
from locked_source import verify, digest, MANIFESTS


def main():
    """仅供显式小测的GPU子进程使用，准确率分母始终是实际评估样本数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--adapted', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--corruption', required=True)
    parser.add_argument('--training_seed', type=int, choices=[42,43,44], required=True)
    parser.add_argument('--max-images', type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_images < 10000:
        parser.error('子集评估只接受1至9999个样本；完整评估使用原入口')
    proof = verify(args.repo)
    protocol = json.loads((MANIFESTS/'formal-protocol.json').read_text())
    if args.corruption not in protocol['corruptions']:
        parser.error('未知扰动')
    sys.path.insert(0, str(args.repo.resolve()))
    # GPU相关导入只在真正执行本评估入口时发生，--help及父进程生成计划均不导入。
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from experiments.complete_evaluate import SharedViews, predict, state_hash
    from dataset import get_transforms
    from model import load_model
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    source = args.repo.resolve()/'data/ModelZoo/Hendrycks2020AugMixWRN.pt'
    if digest(source) != protocol['source_weight_sha256']:
        raise ValueError('源权重摘要不匹配')
    args.out.mkdir(parents=True, exist_ok=False)
    models = {}
    for name, path in [('source',source),('ist',args.adapted)]:
        model = load_model('WRN-40-2', pretrained=False, num_classes=10, device='cuda')
        model.load_state_dict(torch.load(path, map_location='cpu'), strict=True)
        model.eval(); model.requires_grad_(False); models[name] = model
    before = {n: {'state':state_hash(m),'bn':state_hash(m,True)} for n,m in models.items()}
    train, test = get_transforms('cifar10-c')
    dataset = SharedViews(args.repo.resolve()/'data',42,train,test,args.corruption)
    loader = DataLoader(Subset(dataset,range(args.max_images)),batch_size=128,shuffle=False,num_workers=2)
    counts = {n:{'single':0,'eight':0} for n in models};seen=0
    with (args.out/'predictions.csv').open('x',newline='') as handle, torch.inference_mode():
        writer = csv.writer(handle);writer.writerow(['index','label','source_single','source_eight','ist_single','ist_eight'])
        for indices,labels,originals,views in loader:
            if indices.tolist()!=list(range(seen,seen+len(labels))):raise ValueError('评估顺序不符')
            predicted={}
            for name,model in models.items():
                single=predict(model,originals).argmax(-1)
                eight=predict(model,views.flatten(0,1)).reshape(len(labels),8,10).square().flatten(1).argmax(-1)%10
                predicted[name]=(single,eight)
                counts[name]['single']+=(single==labels).sum().item();counts[name]['eight']+=(eight==labels).sum().item()
            for i in range(len(labels)):
                writer.writerow([indices[i].item(),labels[i].item(),*[predicted[n][j][i].item() for n in ['source','ist'] for j in [0,1]]])
            seen+=len(labels)
    after={n:{'state':state_hash(m),'bn':state_hash(m,True)} for n,m in models.items()}
    if seen!=args.max_images or before!=after:raise ValueError('样本数或评估状态检查失败')
    result=dict(count=seen,accuracy={n:{k:v/seen for k,v in c.items()} for n,c in counts.items()},
                before=before,after=after,state_unchanged=True,smoke_only=True,source=proof,
                adaptation_seed=args.training_seed,evaluation_seed=42,subset='原索引0..N-1；非论文完整准确率')
    (args.out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
