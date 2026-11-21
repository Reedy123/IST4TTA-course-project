# 环境材料：已知事实与未解决边界

历史正式与效率实验均使用Linux x86_64、单张RTX 3080 10GB。RTX3060只用于后续文件检查与备份。本发布仅做CPU静态验证，没有安装或重新验证GPU环境。

锁定提交A的`deploy/conda-spec.txt`是实际保留的修复方案：Python3.10.13、pip23.3.1、MKL2021.4.0及其匹配数值库构建，NumPy1.23.5、SciPy1.10.1、FAISS GPU/libfaiss1.7.4、CUDA Toolkit11.4.1。`deploy/requirements.txt`固定torch1.13.1+cu117、torchvision0.14.1+cu117、timm0.6.13等。不要用根目录未固定版本的官方requirements整表替代课程部署方案。

- `formal-pip-freeze.txt`来自180组正式归档，逐字节保留；其中`file:///...`是构建机路径，不是可安装来源，因此不能直接全表pip install。
- `candidate-conda-explicit-linux-64.txt`与`candidate-pip-only-lock.txt`来自已有修复材料。前者含MKL2021.4的明确包URL，与源码修复方案方向一致；仅作为同平台重建候选，未证实所有传递依赖与正式运行完全相同。
- 同批保存的`historical-inconsistent-conda-list.json`仍列MKL2023.1及另一组NumPy/SciPy构建，与explicit不同。这里保留冲突证据，**不能把它与explicit合称为已核验的同一次正式环境锁**。不据此猜测导出时间或替换任何原记录。

未来在Linux实验机上，可从锁定工作树按`deploy/01_install.sh --dry-run`审查方案，再经授权用`--apply`创建独立环境，设置`IST_ENV_PREFIX`。安装、env/weights/data验证和GPU运行都是未来操作，本次未执行。候选explicit只适用Linux x86_64；远端包可能变更或失效，成功重建还需要保存实际conda显式导出、pip来源锁及验证日志。没有随仓库分发包缓存或完整环境，因此不承诺离线/永久可重建。

旧`deploy/README.md`作为锁定历史文件保留，包含阶段性过期说明；不要把其中残留的MKL2023故障建议覆盖这里的2021.4修复版本。新复现仍须通过实际FAISS/PyTorch联合GPU验证才算环境验收。
