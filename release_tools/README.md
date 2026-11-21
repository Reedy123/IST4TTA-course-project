# 独立复现工具

统一入口在仓库根：`python reproduce.py --help`。脚本全部以中文说明作用，默认不启动模型。路径由用户提供或按仓库相对位置计算；没有原服务器绝对路径。

| 子命令/文件 | 作用与默认行为 |
|---|---|
| setup / setup_runtime.py | 从本地固定Git提交准备独立工作树、链接外部数据；拒绝覆盖 |
| weights / fetch_weights.py | 从公开Git历史提取预训练权重到外部目录，SHA核验；不加载模型 |
| install / install_environment.py | 默认打印新环境安装计划，--apply才安装；原环境路径存在即拒绝 |
| check / doctor.py | 默认只读版本诊断；--require-gpu才检查微型联合算子，先检查GPU空闲 |
| inputs / verify_inputs.py | 只读核对所选公开数据/源权重，干净数据完整性仍由官方加载器检查 |
| run / prepare_runs.py | 默认打印新运行计划；--execute才运行，--smoke限制训练/评估范围 |
| adapt_smoke_entry.py | 只给小测的初始test_dataset增加内存子集限制，锁定文件不修改 |
| evaluate_subset.py | 小测最终限量评估，复用原评估函数，校验参数及BN不变 |
| runtime_support.py | 指定设备空闲检查、半秒采样、数值/状态验证，不杀其他进程 |
| summary / summarize_replication.py | 汇总完整的新运行结果，小测与历史正式数据不混合 |
| stats / analyze_results.py | 继续支持对已公开小型180/24历史指标重算，不覆盖原记录 |
| verify_locked.py / locked_source.py | 保留114文件、92源码、提交号及额外源码检查 |

`prepare_runs.make_plan`保留既有API及--modes/--repo/--all-180，新增--mode别名、小测、评估规模、端口和设备。默认Python从当前sys.executable取得，保留虚拟环境符号链接，避免绕开用户已激活环境。--python仍可明确选择其他既有解释器。

小测适应2批、初始/最终各128张；完整运行保留原10000张流程。新包装没有旧16组依赖，也不冒用原总控所有等价审计/续跑证明。每次结果都记录锁定提交、工具摘要、实际命令、设备、样本数和是否小测。其他NVIDIA卡可做功能检查，不标成论文3080硬件。

CPU回归测试：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s release_tools/tests -v
```

测试包含真实的临时Git工作树及公开权重SHA，和明确标注的模拟子进程衔接；不导入torch/faiss，不安装环境、不访问GPU、不改原归档。模拟fixture不能作为真实GPU冒烟结果。未获得GPU运行结果之前，任何“CPU通过”都不代表模型复现成功。
