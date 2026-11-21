"""固定180组清单、交错顺序、仅最终保存和种子42旧评估复用规则。"""
from pathlib import Path
from .final_protocol import MODES, command as stage1_command

CORRUPTIONS = ['gaussian_noise','shot_noise','impulse_noise','defocus_blur','glass_blur',
               'motion_blur','zoom_blur','snow','frost','fog','brightness','contrast',
               'elastic_transform','pixelate','jpeg_compression']
SEEDS = [42,43,44]


def schedule():
    """分四轮每场景一次模式，交错扰动和种子，四模式位置循环平衡。"""
    tasks = []
    modes = list(MODES)
    for round_index in range(4):
        for j in range(15):
            ci = (j*7+round_index*4)%15
            for k in range(3):
                si = (k+j+round_index)%3
                mode = modes[(round_index+ci+si)%4]
                tasks.append(dict(id=f'{CORRUPTIONS[ci]}-s{SEEDS[si]}-{mode}',
                                  corruption=CORRUPTIONS[ci], seed=SEEDS[si], mode=mode,
                                  round=round_index))
    assert len(tasks) == len({t['id'] for t in tasks}) == 180
    return tasks


def command(repo, output, mode, corruption, seed, batches=0):
    """复用既定全部参数，仅替换独立入口、种子和允许的扰动名。"""
    assert corruption in CORRUPTIONS and seed in SEEDS
    cmd = stage1_command(repo, output, mode, 'gaussian_noise', batches)
    cmd[cmd.index(str(Path(repo)/'online_final.py'))] = str(Path(repo)/'online_complete.py')
    cmd[cmd.index('--corruption')+1] = corruption
    cmd[cmd.index('--seed')+1] = str(seed)
    cmd[cmd.index('--master_port')+1] = '16835'
    return cmd
