"""邻居软加权独立入口：保持No-history、FAISS复用及既有训练不变。"""
import os
import hashlib
import json
import numpy as np
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from timm.utils import AverageMeter
from utils.tools import set_seed, Logger, set_logger, reduce_tensor
from model import load_model
from dataset import load_dataset, get_transforms
from utils import UnlabeledDatasetV5, robust_PLCA, MomentumUpdate, MemoryBank
from experiments.admission import Admission
from experiments.input_trace import trace_batch
from experiments.history import select_history, record_quality
from experiments.profiling_runtime import profiler, make_profiled_target
from experiments.weighted_plca import load_weighted
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10-c', help='dataset')
parser.add_argument('--dataset_path', type=str, default='../data/CIFAR10-C/', help='dataset path')
parser.add_argument('--corruption', type=str, default='gaussian_noise', help='corruption')
parser.add_argument('--level', type=int, default=5, help='corruption level')
parser.add_argument('--model', type=str, default='WRN-40-2', help='model')
parser.add_argument('--pretrained', action='store_true', default=False, help='Use pretrained models')
parser.add_argument('--num_classes', type=int, default=7, help='Number of classes')
parser.add_argument('--channels', type=int, default=3, help='Channels of images')
parser.add_argument('--iters', type=int, default=1, help='Number of iterations for every batch')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--max_len', type=int, default=10000, help='Max length of memory bank')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--extend', type=int, default=8)
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--model_load_path', type=str, default='', help='Path to load model')
parser.add_argument('--save_model', action='store_true', default=False, help='Save model or not')
parser.add_argument('--save_path', type=str, default='./results/name/', help='Path to save results')
parser.add_argument('--local_rank', type=int, default=-1, help='node rank for distributed training')
parser.add_argument('--local-rank', type=int, default=-1, help='node rank for distributed training')
parser.add_argument('--admission', choices=['original', 'confidence', 'consistency', 'combined'], default='original')
parser.add_argument('--confidence_threshold', type=float, default=0.8)
parser.add_argument('--consistency_threshold', type=float, default=0.75)
parser.add_argument('--history_mode', choices=['original', 'no-history'], default='no-history')
parser.add_argument('--profile_kind', choices=['diagnostic', 'normal'], required=True)
parser.add_argument('--profile_flow', choices=['no-history', 'hard-only'], required=True)
parser.add_argument('--faiss_mode', choices=['original','reuse-resources'], default='original')
parser.add_argument('--capture_faiss', action='store_true')
parser.add_argument('--audit_faiss', action='store_true')
parser.add_argument('--max_batches', type=int, default=0)
parser.add_argument('--skip_final_eval', action='store_true')
parser.add_argument('--edge_mode',choices=['baseline','edge-c','edge-v','edge-cv','uniform','shuffled-cv'],default='baseline')
args = parser.parse_args()
assert args.faiss_mode == 'reuse-resources' and args.profile_flow == 'no-history'
assert args.history_mode == 'no-history'
profiler.configure(args.profile_kind, args.save_path)
profiled_robust = load_weighted(Path(__file__).parent / 'utils/pseudo_labeling.py',args.edge_mode,.5,args.extend,args.capture_faiss,args.audit_faiss)
robust_PLCA = make_profiled_target(profiled_robust, args.profile_flow, args.extend)
assert args.admission == 'original', '历史贡献实验只允许全量入库'
if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    print(f'RANK and WORLD_SIZE in environ: {rank}/{world_size}')
else:
    rank = -1
    world_size = -1
torch.cuda.set_device(args.local_rank)
dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
dist.barrier()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = args.seed + dist.get_rank()
set_seed(seed)
experiment_name = f'{args.model}_{args.dataset}_{args.corruption}_{args.level}'.replace('/', '-')
if dist.get_rank() == 0:
    logger = Logger(os.path.join(args.save_path, f'{experiment_name}.log'))
    logger.create_config(args)
    logger.info(args)
    set_logger(logger)
model = load_model(args.model, pretrained=args.pretrained, num_classes=args.num_classes, channels=args.channels, device=device)
model_without_ddp = model
model = DDP(model, device_ids=[args.local_rank], broadcast_buffers=False)
if args.model_load_path and os.path.exists(args.model_load_path):
    model_without_ddp.load_state_dict(torch.load(args.model_load_path, map_location='cpu'))
    logger.info(f'Load model from {args.model_load_path}')
    print(f'Load model from {args.model_load_path}')
train_transform, test_transform = get_transforms(args.dataset)
train_dataset = load_dataset(args.dataset, args.dataset_path, [args.corruption, args.level], preprocess=transforms.ToTensor())
test_dataset = load_dataset(args.dataset, args.dataset_path, [args.corruption, args.level], preprocess=test_transform)
train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
param_group = [{'params': model_without_ddp.parameters(), 'lr': args.lr}]
optimizer = torch.optim.SGD(param_group, weight_decay=0.001)
momentum = MomentumUpdate(model)

def criterion(outputs, targets):
    loss = -targets * torch.log_softmax(outputs, dim=-1)
    loss = torch.sum(loss, dim=-1)
    loss = torch.mean(loss)
    return loss
memory_bank = MemoryBank(args.max_len)
admission = Admission(args.admission, args.confidence_threshold, args.consistency_threshold, os.path.join(args.save_path, 'admission.jsonl'))

def compute_pseudo_labels(dataset, preprocess, model, repeat=1, use_memory=True, verbose=True):
    model.eval()

    def model_predict(images, class_names):
        outputs, features = model(images)
        soft_labels = torch.softmax(outputs, dim=-1)
        return (soft_labels, features)
    history_used = select_history(memory_bank, use_memory, repeat, args.history_mode)
    dataset.set_pseudo_labels(define_func=robust_PLCA, model_predict=model_predict, preprocess=preprocess, extend=args.extend, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, bank=history_used, K=50, gamma=3, mode='l2', repeat=repeat, device=device, return_dst=True, verbose=verbose)
    if repeat>0 and (args.capture_faiss or args.audit_faiss):
        pseudo=np.asarray(dataset.get_pseudo_labels());soft=np.asarray(dataset.get_soft_labels())
        if args.audit_faiss:
            row=dict(step=admission.step,pseudo_sha256=hashlib.sha256(pseudo.tobytes()).hexdigest(),soft_sha256=hashlib.sha256(soft.tobytes()).hexdigest(),shape=list(pseudo.shape),dtype=str(pseudo.dtype))
            with open(os.path.join(args.save_path,'pseudo-audit.jsonl'),'a') as f:f.write(json.dumps(row)+'\n')
        if args.capture_faiss:
            path=Path(args.save_path)/'pseudo-inputs';path.mkdir(exist_ok=True)
            np.savez(path/f'{admission.step:03d}.npz',pseudo=pseudo,soft=soft)
    features = dataset.get_features()
    pseudo_labels = dataset.get_pseudo_labels()
    if repeat > 0:
        record_quality(dataset, history_used, os.path.join(args.save_path, 'pseudo_quality.jsonl'), admission.step, args.extend)
    admission.update(memory_bank, features, pseudo_labels, dataset.get_soft_labels(), args.extend, adapt=repeat > 0)
    if verbose:
        print(f'Memory Bank Size: {memory_bank.size()}')
    dist.barrier()
acc_meter = AverageMeter()
model.eval()
with torch.no_grad():
    tbar = tqdm(test_loader, dynamic_ncols=True) if dist.get_rank() == 0 else test_loader
    for images, labels in tbar:
        images = images.to(device)
        labels = labels.to(device)
        outputs, _ = model(images)
        predictions = torch.argmax(outputs, dim=-1)
        acc = torch.sum(predictions == labels) / len(labels)
        acc = reduce_tensor(acc)
        acc_meter.update(acc.item(), len(labels) * dist.get_world_size())
        if dist.get_rank() == 0:
            tbar.set_description(f'Target Testing  [ Accuracy {acc_meter.val:.4f} ({acc_meter.avg:.4f}) ]')
    if dist.get_rank() == 0:
        logger.info(f'Target Dataset [ Accuracy {acc_meter.avg:.4f} ]')
profiler.start()
for batch_idx, batch in enumerate(train_loader):
    profiler.begin_batch(batch_idx, len(batch[0]))
    loss_meter = AverageMeter()
    temp_dataset = UnlabeledDatasetV5(batch, extend=args.extend)
    temp_sampler = torch.utils.data.DistributedSampler(temp_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    temp_loader = DataLoader(temp_dataset, sampler=temp_sampler, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    with profiler.stage('pseudo_other', force=False):
        compute_pseudo_labels(temp_dataset, train_transform, model, repeat=args.repeat)
    trace_batch(batch, temp_dataset, batch_idx, os.path.join(args.save_path, 'input_trace.jsonl'))
    model.train()
    tbar = tqdm(range(args.iters * len(temp_loader)), dynamic_ncols=True) if dist.get_rank() == 0 else None
    with profiler.stage('model_loader_and_loop', force=False):
        for iter in range(args.iters):
            for images, pseudo_labels, soft_labels in temp_loader:
                with profiler.stage('model_update', force=False):
                    optimizer.zero_grad()
                    images = images.to(device)
                    pseudo_labels = pseudo_labels.to(device)
                    soft_labels = soft_labels.to(device)
                    outputs, _ = model(images)
                    loss = criterion(outputs, pseudo_labels)
                    loss += F.kl_div(torch.log_softmax(outputs, dim=-1), soft_labels, reduction='batchmean')
                    loss.backward()
                    optimizer.step()
                    loss_meter.update(loss.item(), images.size(0))
                    if dist.get_rank() == 0:
                        tbar.set_description(f'Target Training [ Batch {batch_idx + 1}/{len(train_loader)} | Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) ]')
                        tbar.update(1)
    if dist.get_rank() == 0:
        tbar.close()
    with profiler.stage('pma', force=False):
        momentum(model, m=0.9)
    with profiler.stage('checkpoint_io', force=True):
        if dist.get_rank() == 0 and args.save_model:
            torch.save(model_without_ddp.state_dict(), os.path.join(args.save_path, f'{experiment_name}_adapt.pth'))
            print(f'Save model to {args.save_path}')
    profiler.end_batch()
    if args.max_batches and batch_idx+1>=args.max_batches:
        break
profiler.stop()
if args.skip_final_eval:
    raise SystemExit(0)
model.eval()
acc_meter = AverageMeter()
tbar = tqdm(train_loader, desc='Testing', dynamic_ncols=True) if dist.get_rank() == 0 else train_loader
for images, labels in tbar:
    temp_dataset = UnlabeledDatasetV5((images, labels), extend=args.extend, verbose=False)
    compute_pseudo_labels(temp_dataset, train_transform, model, repeat=0, verbose=False)
    if dist.get_rank() == 0:
        acc = temp_dataset.acc
        acc_meter.update(acc, len(temp_dataset.samples))
if dist.get_rank() == 0:
    logger.info(f'Target Dataset [ Accuracy {acc_meter.avg:.4f} ]')
    print(f'Target Dataset [ Accuracy {acc_meter.avg:.4f} ]')
