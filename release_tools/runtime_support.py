"""新复现入口的GPU空闲检查、半秒采样和安全进程启动，完全使用标准库。"""
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time


class GpuBusyError(ValueError):
    """保存拒绝启动的现场，只允许无进程低显存的利用率尾迹进入有限复查。"""
    def __init__(self, snapshot, retryable):
        self.snapshot = snapshot
        self.retryable = retryable
        super().__init__('指定GPU未满足空闲条件：' + json.dumps(snapshot, ensure_ascii=False))


def wait_gpu_idle(device=0, audit=None, check=None, timeout=5.0, interval=1.0):
    """保持原阈值，诊断后最多复查5秒；连续两次空闲才放行，实际占用立即停止。"""
    check = check or gpu_idle_info
    start = time.monotonic()
    consecutive = 0
    while True:
        event = {'utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic()-start}
        try:
            snapshot = check(device)
            consecutive += 1
            event.update(status='idle', snapshot=snapshot, consecutive_idle=consecutive)
        except GpuBusyError as error:
            event.update(status='blocked', snapshot=error.snapshot, retryable=error.retryable)
            consecutive = 0
            if not error.retryable:
                if audit:
                    with Path(audit).open('a') as handle:handle.write(json.dumps(event, ensure_ascii=False)+'\n')
                raise
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            event.update(status='unavailable', error=str(error))
            if audit:
                with Path(audit).open('a') as handle:handle.write(json.dumps(event, ensure_ascii=False)+'\n')
            raise
        if audit:
            with Path(audit).open('a') as handle:handle.write(json.dumps(event, ensure_ascii=False)+'\n')
        if consecutive >= 2:
            return snapshot
        remaining = timeout-(time.monotonic()-start)
        if remaining <= 0:
            raise ValueError('GPU有限复查超时，仍未连续两次确认空闲；停止启动')
        time.sleep(min(interval, remaining))


def gpu_idle_info(device=0):
    """只读确认所选设备空闲；不终止其他进程，不自动选择或占用其他GPU。"""
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
    devices = list(csv.reader(io.StringIO(rows), skipinitialspace=True))
    chosen = next((r for r in devices if r and int(r[0]) == device), None)
    if chosen is None:
        raise ValueError('找不到指定GPU：' + str(device))
    index, uuid, name, used, total, utilization = [v.strip() for v in chosen]
    allocation = os.environ.get('CUDA_VISIBLE_DEVICES')
    if allocation is not None:
        allowed = [v.strip() for v in allocation.split(',') if v.strip()]
        if index not in allowed and uuid not in allowed:
            raise ValueError('指定物理GPU不在现有CUDA_VISIBLE_DEVICES分配内；不绕过实验室调度')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    busy = [r for r in csv.reader(io.StringIO(processes), skipinitialspace=True) if r and r[0].strip() == uuid]
    values = [float(used), float(total), float(utilization)]
    if not all(math.isfinite(v) for v in values) or values[0] < 0 or values[1] <= 0 or not 0 <= values[2] <= 100:
        raise ValueError('GPU读数无效，停止启动：' + repr(chosen))
    snapshot = dict(index=int(index), uuid=uuid, name=name, memory_used_MiB=values[0],
                    memory_total_MiB=values[1], utilization_percent=values[2],
                    compute_processes=busy, observed_utc=datetime.now(timezone.utc).isoformat(),
                    paper_hardware=name == 'NVIDIA GeForce RTX 3080')
    if busy or values[0] > 256 or values[2] > 5:
        raise GpuBusyError(snapshot, retryable=not busy and values[0] <= 256)
    return snapshot


def process_environment(device):
    """将单个物理设备映射为子进程cuda:0，避免原监控器固定使用GPU0。"""
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(device), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')


def monitor(command, log, samples, repo, device=0):
    """以半秒间隔采样所选物理GPU；日志只新建，执行失败不清理已有文件。"""
    stop = threading.Event()
    memories = []
    errors = []

    def sample():
        """采样失败不伪造峰值，在返回记录中保留错误。"""
        try:
            with Path(samples).open('x') as handle:
                handle.write('timestamp,memory_MiB,utilization_percent\n')
                while not stop.is_set():
                    result = subprocess.run(['nvidia-smi', '-i', str(device), '--query-gpu=timestamp,memory.used,utilization.gpu',
                                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        handle.write(result.stdout)
                        handle.flush()
                        for row in csv.reader(io.StringIO(result.stdout)):
                            if len(row) == 3:
                                memories.append(float(row[1]))
                    else:
                        errors.append('nvidia-smi采样失败')
                    stop.wait(.5)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            errors.append(str(error))

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    start = time.monotonic()
    try:
        with Path(log).open('x') as handle:
            result = subprocess.run(command, cwd=repo, env=process_environment(device),
                                    stdout=handle, stderr=subprocess.STDOUT)
    finally:
        stop.set()
        thread.join()
    return dict(exit_code=result.returncode, launcher_seconds=time.monotonic()-start,
                peak_sampled_gpu_MiB=max(memories) if memories else None,
                sample_interval_seconds=.5, sampling_errors=errors, physical_gpu=device)


def check_metrics(metrics, expected_images):
    """只接受样本数正确、准确率有限和评估状态未更新的结果。"""
    if (metrics.get('count') != expected_images or metrics.get('state_unchanged') is not True
            or 'before' not in metrics or metrics['before'] != metrics.get('after')):
        raise ValueError('评估样本数不符或参数/BN发生变化')
    for mode in ['source', 'ist']:
        for view in ['single', 'eight']:
            value = metrics['accuracy'][mode][view]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('准确率不是有效的[0,1]数值')
