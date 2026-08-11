# Mini-FORGE

一个面向学习与研究的轻量 Semantic ID 生成式推荐实现：

```text
MovieLens 1M（用户自行下载）
  → 正反馈过滤、迭代 K-core、时序切分
  → MiniLM 电影内容向量
  → 三级 RQ-VAE 残差量化
  → 唯一 Semantic ID
  → Mini-T5 自回归预测
  → Trie 约束 Beam Search
  → SID 回表与 Item 级评测
```

> 本项目是独立的教学/研究实现，概念上受到 FORGE 工作启发，并非 FORGE 作者团队的官方实现，也不代表其背书。论文和官方仓库见[致谢与引用](#致谢与引用)。

## 功能

- MovieLens 1M 下载、格式校验和逐用户时序切分；
- `rating >= 4` 正反馈过滤和迭代 K-core；
- 基于训练时间段构造滑动前缀样本与 i2i 共现对；
- 使用 `sentence-transformers/all-MiniLM-L6-v2` 生成冻结内容向量；
- 三级 RQ-VAE、K-means 码本初始化及内容/协同联合训练；
- SID 碰撞编号、唯一回表和码本质量评估；
- 从头初始化的小型 T5，不下载 T5 预训练权重；
- 有效 SID Trie 约束 Beam Search，并支持去除已看电影；
- HR、NDCG、MRR、Catalog Coverage 和生成质量指标；
- 不依赖 MovieLens 和外部模型的 CPU smoke 测试。

## 环境

- Python 3.10–3.12；
- PyTorch 2.2+；
- 正式配置建议使用支持 CUDA 的 NVIDIA GPU；
- smoke 配置可以仅使用 CPU。

CUDA 版 PyTorch 请先按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)安装，再安装项目：

```bash
git clone <your-repository-url>
cd mini-forge
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Linux/macOS：

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

检查安装：

```bash
mini-forge --help
```

## 先运行 smoke 流程

Smoke 数据是脚本生成的合成数据，只用于检查代码链路，不代表正式实验效果：

```bash
python scripts/create_smoke_data.py
mini-forge --config configs/smoke.yaml prepare
mini-forge --config configs/smoke.yaml embed
mini-forge --config configs/smoke.yaml train-rqvae
mini-forge --config configs/smoke.yaml export-sid
mini-forge --config configs/smoke.yaml train-generator
mini-forge --config configs/smoke.yaml eval-generator --split test
```

运行单元测试：

```bash
python -m unittest discover -s tests -v
```

## 正式流程

在下载 MovieLens 1M 前，请先阅读[数据与第三方许可](DATA_LICENSE.md)。MovieLens 1M 的原始许可限制再分发，并限制未经许可的商业/营利用途。

```bash
mini-forge download
mini-forge prepare
mini-forge embed
mini-forge train-rqvae
mini-forge export-sid
mini-forge train-generator
mini-forge eval-generator --split test
```

也可以一条命令执行核心链路：

```bash
mini-forge run-all
```

已有合法获取的原始数据时：

```bash
mini-forge run-all --skip-download
```

默认配置位于 `configs/default.yaml`。命令行支持覆盖任意配置项：

```bash
mini-forge --set rqvae.codebook_size=32 train-rqvae
mini-forge --set generator.batch_size=8 train-generator
mini-forge --set inference.num_beams=10 eval-generator --split validation
```

## 目录结构

```text
mini_forge/
├── configs/
│   ├── default.yaml        # MovieLens 1M 正式配置
│   └── smoke.yaml          # CPU 合成数据配置
├── scripts/
│   └── create_smoke_data.py
├── src/mini_forge/
│   ├── data.py             # 下载、K-core、切分与训练样本
│   ├── content.py          # MiniLM/Hashing 内容向量
│   ├── rqvae.py            # RQ-VAE 与训练
│   ├── sid.py              # SID 导出、碰撞处理与质量指标
│   ├── generator.py        # Mini-T5、Trie、训练与生成评测
│   ├── metrics.py          # Item 级排名指标
│   └── cli.py              # 命令行入口
├── tests/
├── DATA_LICENSE.md
├── THIRD_PARTY_NOTICES.md
├── OPEN_SOURCE_CHECKLIST.md
└── LICENSE
```

运行时会生成 `data/` 和 `artifacts/`。

| 产物 | 默认路径 |
|---|---|
| 处理后序列和 i2i 对 | `data/processed/` |
| 内容 embedding | `artifacts/content/` |
| RQ-VAE checkpoint | `artifacts/rqvae/best.pt` |
| MovieID-SID 映射 | `artifacts/sid/movie_to_sid.json` |
| SID 质量报告 | `artifacts/sid/quality_metrics.json` |
| Mini-T5 checkpoint | `artifacts/generator/sid/best.pt` |
| 生成评测结果 | `artifacts/generator/sid/test_metrics.json` |

## 许可

本仓库自有代码采用 [Apache License 2.0](LICENSE)。该许可不覆盖 MovieLens 数据、第三方模型、第三方库或用户自行训练的衍生产物；详见 [DATA_LICENSE.md](DATA_LICENSE.md) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 致谢与引用

项目在研究思路上受到以下工作的启发：

- FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets，
  [论文](https://arxiv.org/abs/2509.20904)，[官方代码](https://github.com/selous123/al_sid)。

如果该工作对你的研究有帮助，请引用原论文，并遵守数据和第三方依赖各自的许可。

```bibtex
@article{fu2025forge,
  title={FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets},
  author={Fu, Kairui and Zhang, Tao and Xiao, Shuwen and Wang, Ziyang and Zhang, Xinming and Zhang, Chenchi and Yan, Yuliang and Zheng, Junjun and others},
  journal={arXiv preprint arXiv:2509.20904},
  year={2025}
}
```
