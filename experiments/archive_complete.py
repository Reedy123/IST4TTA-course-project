"""全部180组结束后校验并归档源码、日志、论文表格和最终权重，不包含数据集。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import zipfile
from .final_protocol import digest
from .run_final_suite import source_inventory, write


def main(root):
    """生成可在Mac独立验证的SHA256清单和ZIP，保留所有历史尝试。"""
    repo=Path(__file__).resolve().parents[1]
    assert json.loads((root/'all-complete.json').read_text())['groups']==180
    assert json.loads((root/'statistics.json').read_text())['groups']==180
    protocol=json.loads((root/'protocol.json').read_text())
    assert source_inventory(repo)==protocol['source_files']
    with tarfile.open(root/'source-snapshot.tar.gz') as tar:
        for name,sha in protocol['source_files'].items():
            assert hashlib.sha256(tar.extractfile(name).read()).hexdigest()==sha
    diff=subprocess.check_output(['git','diff','--','online.py','utils','model','dataset','scripts'],cwd=repo)
    assert diff==b''
    (root/'official-source.diff').write_bytes(diff)
    for row in json.loads((root/'results.json').read_text()):
        assert digest(Path(row['checkpoint']))==row['checkpoint_sha256']
        assert digest(Path(row['predictions']))==row['predictions_sha256']
    validation=root/'validation';validation.mkdir(exist_ok=True)
    smoke=Path(str(root)+'-smoke')
    for src,name in [(smoke/'complete.json','smoke-complete.json'),
                     (Path.home()/'work/ist-complete-unit-20261117.log','unit-tests.log'),
                     (Path.home()/'work/ist-complete-data-verification.json','data-verification.json'),
                     (Path(str(root)+'-suite.log'),'suite.log')]:
        shutil.copy2(src,validation/name)
    target=root/'backup-final.zip'
    if target.exists():
        print(json.dumps(dict(path=str(target),sha256=digest(target),bytes=target.stat().st_size)),flush=True)
        return
    files=[]
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.name!='SHA256-FINAL.json' and p.suffix not in ['.zip','.part']:
            files.append(dict(file=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p)))
    manifest=root/'SHA256-FINAL.json'
    write(manifest,dict(files=files,successful_final_weights=180))
    partial=root/'backup-final.zip.part'
    # 上次归档若中断，保留部分包并另起编号，禁止覆盖。
    number=1
    while partial.exists():
        partial=root/f'backup-final-{number}.zip.part';number+=1
    with zipfile.ZipFile(partial,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for r in files:z.write(root/r['file'],r['file'])
        z.write(manifest,manifest.name)
    with zipfile.ZipFile(partial) as z:assert z.testzip() is None
    partial.rename(target)
    receipt=dict(path=str(target),bytes=target.stat().st_size,sha256=digest(target),files=len(files),weights=180)
    write(Path(str(root)+'-archive-receipt.json'),receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    main(args.root)
