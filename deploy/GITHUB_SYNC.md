# GitHub 同步方案（尚未提交或推送）

用途：仅同步复现工具和小型检查记录，数据及权重通过文件上传单独传输。

建议创建个人部署工具仓库，仅将本工具包复制进去；不要在官方仓库提交修改，不要对整个工作目录执行 git add .。

提交白名单：README.md、GITHUB_SYNC.md、PROVENANCE.txt、*.sh、*.py、conda-spec.txt、constraints.txt、requirements.txt、package-evidence.json。

不提交：官方仓库副本、数据、模型、压缩包、环境、运行结果、.env、SSH 密钥或访问令牌。工具包无需任何凭据。

新个人仓库根目录 .gitignore 建议：

```gitignore
IST4TTA-official/
ist-data/
data/
downloads/
runtime/
deployment-record/
results*/
__pycache__/
*.npy
*.pt
*.pth
*.tar
*.gz
*.zip
*.part
.env*
*.pem
*.key
.DS_Store
```

提交前执行 git diff --cached --stat 和 git diff --cached，并检查 git ls-files；明确确认没有大型文件或凭据再提交。模型即使只有 8.6MB 也不提交。官方仓库在服务器按 PROVENANCE.txt 固定 commit 获取，权重可由官方 clone 获取，个人仓库不重复托管。

组合布局：

```text
ist-work/
  IST4TTA-official/         # 固定 commit 的原仓库，含官方权重
  ist-minimal-kit/          # 本工具包
  ist-data/                # 校验后的数据和压缩包，可只上传 data/
  env-ist310/              # 未来服务器创建
  results-gaussian5-run1/  # 未来正式实验输出
```

上传前必须得到 ist-data/verified-manifest.json。将 ist-data/data/ 下的两个目录复制到 IST4TTA-official/data/；不要多嵌套一层 data。再依次运行 02_verify.py 的 env、weights、data 阶段。若无需服务器保留原始包，只上传已校验的 data/ 和清单，可节省约 3GB 传输。

Mac 恢复下载、校验及整理（使用已有 Python3/NumPy，不安装依赖）：

```bash
python3 /绝对路径/ist-minimal-kit/04_prepare_data.py --base /绝对路径/ist-data --download
```

下载后可在 Mac 使用相同脚本但不加 --download 重新校验。任何断言失败都不能上云作为正式数据。
