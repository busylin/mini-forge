# 开源发布检查清单

## 应上传

- `.gitignore`；
- `README.md`、`LICENSE`、`NOTICE`；
- `DATA_LICENSE.md`、`THIRD_PARTY_NOTICES.md`；
- `pyproject.toml`、`requirements.txt`；
- `configs/default.yaml`、`configs/smoke.yaml`；
- `scripts/create_smoke_data.py`；
- `src/mini_forge/` 中公开版的数据、内容编码、RQ-VAE、SID、Mini-T5、Trie、指标、配置和 CLI 代码；
- `tests/test_core.py`。

## 不应上传

- 原始 MovieLens 压缩包和 `ratings.dat`、`movies.dat`、`users.dat`；
- `data/` 下的全部处理结果；
- `artifacts/` 下的 embedding、SID 映射、checkpoint、预测结果和指标；
- SASRec、HSTU、门控融合、辅助任务等私有实验代码；
- 实验计划、实验记录、私有方案和本机训练日志；
- `.venv*`、`__pycache__`、`*.egg-info`；
- `.env`、访问令牌、密钥和含个人绝对路径的配置。

## 发布前执行

```bash
python -m unittest discover -s tests -v
git status --short
git ls-files
```

逐项确认 `git ls-files` 中没有 `data/`、`artifacts/`、模型权重、JSONL 用户序列、实验文档和密钥文件。首次推送前还应在 GitHub 仓库页面确认默认分支中不存在大文件。

如果敏感文件曾进入 Git 历史，仅新增 `.gitignore` 不会将它移除；必须在公开前清理 Git 历史或重新创建一个干净仓库。

