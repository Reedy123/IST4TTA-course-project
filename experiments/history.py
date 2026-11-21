"""控制适应阶段历史读取，并记录只用于离线分析的伪标签质量。"""
import json
from pathlib import Path
import numpy as np


def select_history(bank, use_memory=True, repeat=1, mode='original'):
    """no-history仅切断适应时的历史读取，原版和最终评估路径保持原样。"""
    assert mode in ['original', 'no-history']
    if not use_memory or (mode == 'no-history' and repeat > 0):
        return [], []
    return bank.get_all()


def record_quality(dataset, history, destination, step, views):
    """真值仅计算诊断指标，不修改特征、伪标签或训练选择。"""
    soft = np.asarray(dataset.get_soft_labels())
    corrected = np.asarray(dataset.get_pseudo_labels())
    truth = np.asarray([int(value) for value in dataset.labels])
    assert soft.shape == corrected.shape and len(soft) == len(truth)*views
    classes = soft.shape[-1]
    initial_view = soft.argmax(-1)
    final_view = corrected.argmax(-1)
    repeated_truth = np.repeat(truth, views)
    initial_group = soft.reshape(len(truth), views*classes).argmax(-1) % classes
    final_group = (soft*corrected).reshape(len(truth), views*classes).argmax(-1) % classes
    row = dict(step=step, groups=len(truth), views=len(soft), history_entries_read=len(history[0]),
               raw_group_correct=int((initial_group==truth).sum()), corrected_group_correct=int((final_group==truth).sum()),
               raw_view_correct=int((initial_view==repeated_truth).sum()), corrected_view_correct=int((final_view==repeated_truth).sum()),
               group_wrong_to_right=int(((initial_group!=truth)&(final_group==truth)).sum()),
               group_right_to_wrong=int(((initial_group==truth)&(final_group!=truth)).sum()),
               view_label_changes=int((initial_view!=final_view).sum()), diagnostic_only=True)
    with Path(destination).open('a') as file:
        file.write(json.dumps(row)+'\n')
