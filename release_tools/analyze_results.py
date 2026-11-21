"""用CPU从完整180/24原始JSON重新统计，只写入全新的输出目录。"""
import argparse
import csv
import json
import statistics as st
from pathlib import Path
import sys
from locked_source import verify, digest


def write_csv(path, rows):
    """写入新目录中的CSV，保留未舍入数值。"""
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(repo, formal_path, efficiency_path, output):
    """校验组合完整性后复用原正式统计器，效率按全部12对逐对汇总。"""
    proof = verify(repo)
    formal = json.loads(formal_path.read_text())
    efficiency = json.loads(efficiency_path.read_text())
    protocol = json.loads((Path(__file__).resolve().parents[1] / 'reproduction/manifests/formal-protocol.json').read_text())
    expected = {(c, s, m) for c in protocol['corruptions'] for s in [42, 43, 44] for m in protocol['modes']}
    if len(formal) != 180 or {(r['corruption'], r['seed'], r['mode']) for r in formal} != expected:
        raise ValueError('正式记录必须完整覆盖15扰动×3适应随机种子×4模式，共180条且无重复')
    corruptions = ['gaussian_noise', 'motion_blur', 'fog', 'jpeg_compression']
    expected_eff = {(c, k, m) for c in corruptions for k in range(3) for m in ['Original', 'Ours']}
    if len(efficiency) != 24 or {(r['corruption'], r['repeat'], r['mode']) for r in efficiency} != expected_eff:
        raise ValueError('效率记录必须完整覆盖4扰动×3计时重复×2模式，共24条且无重复')
    if any(r['legacy_timing_reused'] for r in formal):
        raise ValueError('不允许将旧16组探索计时混入正式结果')
    for row in efficiency:
        if not row.get('weights_exact', False):
            raise ValueError('效率记录缺少与正式权重一致的校验标记')
    output = output.expanduser().resolve()
    if output.exists():
        raise ValueError('输出目录已存在，拒绝覆盖原始数据或先前统计')
    source_paths = [formal_path.resolve(), efficiency_path.resolve(), Path(repo).resolve()]
    if any(output == p or output in p.parents or p in output.parents for p in source_paths):
        raise ValueError('输出必须位于输入文件/锁定仓库之外的独立新目录')
    # 该模块及其导入链只有标准库；原实现的torch导入仅在未调用的权重检查函数内部。
    sys.path.insert(0, str(Path(repo).resolve()))
    from experiments.summarize_complete import summarize
    pairs = []
    summary = []
    indexed = {(r['corruption'], r['repeat'], r['mode']): r for r in efficiency}
    for corruption in corruptions:
        for repeat in range(3):
            a, b = [indexed[corruption, repeat, m]['adaptation_seconds'] for m in ['Original', 'Ours']]
            if a <= 0 or b <= 0:
                raise ValueError('适应耗时必须为正')
            pairs.append(dict(corruption=corruption, repeat=repeat + 1, original_seconds=a,
                              ours_seconds=b, speedup=a / b, reduction_percent=100 * (1 - b / a)))
        for mode in ['Original', 'Ours']:
            rows = [indexed[corruption, k, mode] for k in range(3)]
            summary.append(dict(corruption=corruption, mode=mode, n=3,
                                adaptation_mean=st.mean(r['adaptation_seconds'] for r in rows),
                                adaptation_sd=st.stdev(r['adaptation_seconds'] for r in rows),
                                faiss_mean=st.mean(r['faiss_seconds'] for r in rows),
                                faiss_sd=st.stdev(r['faiss_seconds'] for r in rows)))
    output.mkdir(parents=True, exist_ok=False)
    summarize(output, formal, complete=True)
    write_csv(output / 'efficiency-pairs.csv', pairs)
    write_csv(output / 'efficiency-summary.csv', summary)
    report = dict(source=proof, formal_records=180, efficiency_records=24, pairs=12,
                  ddof=1, speedup_mean=st.mean(r['speedup'] for r in pairs),
                  speedup_sd=st.stdev(r['speedup'] for r in pairs),
                  reduction_mean=st.mean(r['reduction_percent'] for r in pairs),
                  reduction_sd=st.stdev(r['reduction_percent'] for r in pairs),
                  slower_pairs=[r for r in pairs if r['speedup'] < 1],
                  formal_results_sha256=digest(formal_path), efficiency_results_sha256=digest(efficiency_path),
                  accuracy_note='24组准确率是复用评估，不计为新评估；不与180组合并',
                  implementation='原正式统计器+新发布效率统计包装；从未用于生成历史模型')
    (output / 'analysis-audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


def main():
    """解析显式路径，统计前确认所有必需输入存在。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--locked-repo', type=Path, required=True)
    parser.add_argument('--formal-results', type=Path, required=True)
    parser.add_argument('--efficiency-results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    missing = [str(p) for p in [args.formal_results, args.efficiency_results] if not p.is_file()]
    if missing:
        parser.error('缺少输入文件：' + ', '.join(missing))
    try:
        print(json.dumps(analyze(args.locked_repo, args.formal_results, args.efficiency_results, args.output), ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f'统计失败：{error}\n')


if __name__ == '__main__':
    main()
