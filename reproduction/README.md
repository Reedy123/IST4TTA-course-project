# 复现资料与版本边界

本目录是提交A之后新增的发布材料，没有参与历史模型训练或计时。原始实验源码固定在标签`paper-experiments-180-24-locked`，其完整114个文件和92份源码摘要分别见`manifests/snapshot-files.json`、`source-files.json`。准确提交号、原快照摘要及材料摘要见`manifests/provenance.json`。

- `original/efficiency24_runner.py`：原24组执行器的逐字节副本，保留原路径、恢复逻辑、3080检查和权重/输入轨迹一致性断言。
- `original/build_report.py`：生成原效率配对CSV、汇总CSV及报告的原脚本，逐字节保留；含固定相对路径和写入操作，不要直接对原归档运行。
- `manifests/formal-protocol.json`与`efficiency24-plan.json`：原协议原样保留，包括历史路径和当时的术语。路径是证据，不能在原协议中改成当前路径。文档中“适应随机种子”为规范表述。
- `reference-results/`：原有小型指标JSON/CSV/统计JSON的原样副本，仅含指标、摘要和历史文件位置；不含图像、模型参数、预测明细大包或环境目录。可在CPU上重新统计，无需GPU/数据集/权重。路径字段指向当时服务器，不应据此声称模型文件也已发布。
- `environment/`：锁定版本环境说明、候选包导出及冲突说明，详见该目录README。
- `../release_tools/`：发布阶段的新工具，默认只读核验或生成命令；与上面原始执行器明确区分。

## 原24组流程的限制

原执行器固定`~/work/IST4TTA-course-project`、`~/work/ist-complete-20261117-run02`、`~/work/envs/ist310`以及恢复上传目录。它检查正式权重逐张量相同并复用其准确率，还检查输入、FAISS、伪标签轨迹。它需要历史完整正式归档，不能仅依赖这里的小型指标JSON运行；缺失历史权重时不能把检查删掉后仍声称复现原24组。

原执行器应放在锁定工作树之外，并明确`PYTHONPATH`和工作目录指向锁定仓库。只保留文件供审阅；本次未执行原执行器，也没有把其上传/解压恢复逻辑改造成新的实验方法。新地点、新环境运行前须逐项核实这些路径和依赖；如另写包装，标注为新复现，不覆盖原文件。

## 原180组总控的限制

`experiments/run_complete_suite.py`和`start_complete_tmux.sh`依赖旧16组正式第一阶段记录及匹配源码的smoke证明；`verify_complete_data.py`还依赖旧数据校验记录。这些是原历史流程，不是从零入口。本发布提供`release_tools/prepare_runs.py`从同一锁定算法生成新的独立运行计划，且可显式启动新运行；它不复用旧16组准确率、不提供历史续跑兼容、不自动宣称完成原总控全部审计。

## 源码锁与README检查

新增任意.py/.sh都会改变`source_inventory()`；修改根README会触发旧验证器的官方文件一致性检查。不要在发布工作树中关闭这些检查。请单独检出A标签，使用发布目录中的外部工具检查/运行；锁定工作树保留原README。由此既可保留易读课程首页，又不修改历史算法与验证器。
