"""用标准库核验独立实验工作树；不导入模型、不改写被检查文件。"""
import hashlib
import json
from pathlib import Path
import subprocess

MANIFESTS = Path(__file__).resolve().parents[1] / 'reproduction' / 'manifests'


def digest(path):
    """分块计算文件SHA-256，避免整份大型数据占用内存。"""
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify(repo):
    """同时检查114个文件、92份源码清单及提交，不接受发布分支替代锁定版本。"""
    repo = Path(repo).expanduser().resolve()
    expected = json.loads((MANIFESTS / 'snapshot-files.json').read_text())
    source = json.loads((MANIFESTS / 'source-files.json').read_text())
    provenance = json.loads((MANIFESTS / 'provenance.json').read_text())
    missing = [name for name in expected if not (repo / name).is_file()]
    changed = [name for name, checksum in expected.items()
               if (repo / name).is_file() and digest(repo / name) != checksum]
    if missing or changed:
        raise ValueError(f'锁定源码不匹配；缺失：{missing}；内容不同：{changed}')
    command = ['git', '--no-optional-locks', '-C', str(repo)]
    commit = subprocess.check_output(command + ['rev-parse', 'HEAD'], text=True).strip()
    if commit != provenance['locked_commit']:
        raise ValueError('请检出锁定标签，不能在发布工具/README分支上运行历史验证器')
    names = subprocess.check_output(command + ['ls-files', '-co', '--exclude-standard'], text=True).splitlines()
    actual = {name: digest(repo / name) for name in set(names)
              if (repo / name).is_file() and name.endswith(('.py', '.sh')) and not name.startswith('data/')}
    if actual != source:
        raise ValueError('92份源码清单发生变化；请将发布工具和24组执行器放在锁定工作树之外')
    return {'commit': commit, 'snapshot_files': len(expected), 'source_files': len(actual), 'verified': True}
