"""从已克隆的公开上游Git对象提取预训练权重，SHA核验后只写外部数据目录。"""
import argparse
import json
from pathlib import Path
import subprocess
from locked_source import digest, MANIFESTS
from setup_runtime import RELEASE


SOURCE_COMMIT='03ba8e716f9bb6e0a586718f8836362b7090692b'
SOURCE_FILE='data/ModelZoo/Hendrycks2020AugMixWRN.pt'


def prepare_weights(data_dir):
    """优先使用随原历史保留的公开模型，已有文件只核验，不覆盖、不加载模型。"""
    data_dir=Path(data_dir).expanduser().resolve()
    if not data_dir.is_dir() or data_dir==RELEASE or RELEASE in data_dir.parents:
        raise ValueError('请使用仓库外已存在的数据目录')
    checksum=json.loads((MANIFESTS/'provenance.json').read_text())['source_weight_sha256']
    target=data_dir/'ModelZoo/Hendrycks2020AugMixWRN.pt'
    if target.exists():
        if digest(target)!=checksum:raise ValueError('已有权重摘要不符，拒绝覆盖')
        return dict(path=str(target),sha256=checksum,already_present=True)
    result=subprocess.run(['git','-C',str(RELEASE),'show',SOURCE_COMMIT+':'+SOURCE_FILE],capture_output=True)
    if result.returncode:
        raise ValueError('缺少上游历史对象；请完整git clone，或按文档从固定官方URL获取并核验')
    import hashlib
    if hashlib.sha256(result.stdout).hexdigest()!=checksum:raise ValueError('公开Git模型对象摘要不符')
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as handle:handle.write(result.stdout)
    return dict(path=str(target),sha256=checksum,bytes=len(result.stdout),already_present=False,source_commit=SOURCE_COMMIT)


def main():
    """准备公开源权重，不下载新模型、不读取私钥或实验权重。"""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data-dir',type=Path,required=True)
    args=parser.parse_args()
    try:print(json.dumps(prepare_weights(args.data_dir),ensure_ascii=False,indent=2))
    except (ValueError,OSError) as error:parser.exit(2,f'源权重准备失败：{error}\n')


if __name__=='__main__':main()
