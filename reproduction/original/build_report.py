"""从原始记录重算论文统计，并导出中文报告和逐组数据。"""
import json,csv,statistics as s,itertools,shutil
from pathlib import Path
out=Path('outputs/ist-final-20261120'); e=json.loads(Path('work/ist-efficiency24-20261118-run01/results.json').read_text()); f=json.loads(Path('work/ist-complete-20261117-run02/results.json').read_text())
def ms(v):
 """返回均值与样本标准差。"""
 return f'{s.mean(v):.2f} ± {s.stdev(v):.2f}'
def export(name,rows):
 """导出平铺统计表，方便后续论文排表。"""
 with (out/name).open('w') as h:
  w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
cs=['gaussian_noise','motion_blur','fog','jpeg_compression']; pairs=[];summary=[]
for c in cs:
 for k in range(3):
  a,b=[next(r for r in e if (r['corruption'],r['repeat'],r['mode'])==(c,k,m)) for m in ['Original','Ours']]
  pairs.append(dict(corruption=c,repeat=k+1,original_seconds=a['adaptation_seconds'],ours_seconds=b['adaptation_seconds'],speedup=a['adaptation_seconds']/b['adaptation_seconds'],reduction_percent=100*(1-b['adaptation_seconds']/a['adaptation_seconds'])))
 for m in ['Original','Ours']:
  rs=[r for r in e if r['corruption']==c and r['mode']==m]
  summary.append(dict(corruption=c,mode=m,n=len(rs),adaptation_mean=s.mean(r['adaptation_seconds'] for r in rs),adaptation_sd=s.stdev(r['adaptation_seconds'] for r in rs),faiss_mean=s.mean(r['faiss_seconds'] for r in rs),faiss_sd=s.stdev(r['faiss_seconds'] for r in rs)))
export('efficiency-pairs.csv',pairs);export('efficiency-summary.csv',summary)
export('efficiency-runs.csv',[{k:r[k] for k in ['corruption','repeat','mode','adaptation_seconds','faiss_seconds','peak_MiB','weights_exact','checkpoint_sha256']} for r in e])
export('formal-runs.csv',[{k:r[k] for k in ['id','corruption','seed','mode','single_percent','eight_percent','adaptation_seconds','faiss_seconds','peak_sampled_gpu_MiB']} for r in f])
expected={(c,k,m) for c in cs for k in range(3) for m in ['Original','Ours']};actual={(r['corruption'],r['repeat'],r['mode']) for r in e}
lines=['# IST最终实验数据及分析报告','',f'24组清单：{len(e)}/24；缺失组：{sorted(expected-actual)}；重复条目：{len(e)-len(actual)}。正式记录：{len(f)}/180。','', '## 24组效率稳定性','', 'RTX 3080；4扰动×3次计时重复×2模式，固定训练种子42。标准差均为样本标准差（ddof=1）。每个扰动只有3对，不能当作3个独立训练种子。','', '|扰动|Original适应秒|Ours适应秒|配对时间减少%|配对加速倍数|','|---|---:|---:|---:|---:|']
for c in cs:
 a=[[r['adaptation_seconds'] for r in e if r['corruption']==c and r['mode']==m] for m in ['Original','Ours']];p=[r for r in pairs if r['corruption']==c]
 lines.append(f"|{c}|{ms(a[0])}|{ms(a[1])}|{ms([r['reduction_percent'] for r in p])}|{ms([r['speedup'] for r in p])}|")
lines+=['',f"12对总体时间减少：{ms([r['reduction_percent'] for r in pairs])}%；配对加速：{ms([r['speedup'] for r in pairs])}倍。这是4类场景的描述性汇总，不代表12个独立训练种子。",f"所有Original运行适应秒：{ms([r['adaptation_seconds'] for r in e if r['mode']=='Original'])}；Ours：{ms([r['adaptation_seconds'] for r in e if r['mode']=='Ours'])}。",'', '定义：配对加速倍数=T_Original/T_Ours；配对时间减少=100×(1−T_Ours/T_Original)。先逐对计算，再求均值与标准差，不能用均值之比替代。','', '逐对数据见efficiency-pairs.csv，FAISS统计见efficiency-summary.csv。','', '高斯噪声计时波动明显，不能声称每次均加速，也不能据此声称普遍稳定或统计非劣效。保留所有重复，不剔除慢样本。','', '## 180组正式实验','', '15扰动×3训练种子（42/43/44）×4模式。总体先对每个种子平均15扰动，再统计3种子。24组仅作为计时稳定性补充，不加入180组准确率或耗时均值。','', '|模式|单视图%|八视图%|适应秒|FAISS秒|','|---|---:|---:|---:|---:|']
for m in ['Original','Original-Reuse','No-history','Ours']:
 vals={k:[s.mean(r[k] for r in f if r['mode']==m and r['seed']==seed) for seed in [42,43,44]] for k in ['single_percent','eight_percent','adaptation_seconds','faiss_seconds']}
 lines.append('|'+m+'|'+'|'.join(ms(v) for v in vals.values())+'|')
lines+=['','正式结论：Ours主要改善计算效率，准确率总体约下降0.03个百分点。资源复用保持对应准确率；取消历史读取改变算法行为，不能将Ours描述为对Original完全数值等价。','', '24组准确率复用正式seed42评估；运行器记录每次与正式权重逐张量一致，同时检查输入和审计轨迹。当前复核还需结合文件摘要及备份校验结果，不把复用准确率写成新评估。','', '固定八视图评估沿用概率平方后跨8×10最大值，无PLCA最终评估；适应计时含取数、日志、PMA、最终一次保存，不含初始测试和最终评估。显存为半秒设备采样峰值，不是精确分配峰值。旧16组仅复用经验证准确率，不混入旧计时。','', '四个开发扰动不能称为完全未见验证；仅三训练种子，不宣称统计非劣效。两套结果均来自RTX 3080，本轮RTX 3060只用于读取和备份。','', '## 完整性与退还条件','', '文件完整性详见remote-integrity.json与local-integrity.json。只有完整下载、远端和本地摘要一致、两个ZIP通过CRC检查、关键文件核验无缺失后，才可确认本次实验资产可安全离开实例。']
(out/'最终分析报告.md').write_text('\n'.join(lines));shutil.copy('work/ist-complete-20261117-run02/PAPER_TABLES.md',out/'正式实验论文表.md')
print('\n'.join(lines[:18]))
