# Mini-FORGE

一个面向学习与研究的轻量 Semantic ID 生成式推荐实现：

```text
MovieLens 1M（用户自行下载）
  → 正反馈过滤、迭代 K-core、时序切分
  → MiniLM 电影内容向量
  → 三级 RQ-VAE 残差量化
  → 唯一 Semantic ID
  → 用户属性前缀 + 历史 SID 序列
  → Mini-T5 自回归预测下一部电影的 SID
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
- 编码 UserID、Gender、Age、Occupation，并使用结构化用户前缀；
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

## 用户信息与训练数据

`prepare` 读取 `ratings.dat`、`movies.dat` 和 `users.dat`，输出包含四类用户字段的 `train.jsonl`、`validation.jsonl`、`test.jsonl`，以及 `users.csv`。下面是格式示例（非随仓库发布的真实用户数据）：

```json
{"user_id": 11, "gender": "M", "age": "18", "occupation": "15", "history": [101, 102], "target": 103}
```

- 先过滤正反馈并执行 K-core，再按每位用户的交互时间排序。
- 最后两个物品分别作为验证目标和测试目标；训练样本仅从之前的交互构造滑动前缀。
- 默认历史最短5部、最长50部；测试历史包含验证物品，但不包含测试目标。
- `age` 和 `occupation` 保留 MovieLens 的年龄组、职业类别编码，并不是精确年龄或职业名称；ZIP code 不输入模型。
- 合成 smoke 数据没有 `users.dat` 时，Gender/Age/Occupation 使用 `UNK`，UserID 仍保留。若提供了 `users.dat` 却缺少某个保留用户的记录，预处理会明确报错。

JSONL 中的 `history`、`target` 保存电影 ID。`SIDDataset` 在读取时通过 `movie_to_sid.json` 将其转换为 SID，实际编码器输入为：

```text
<USER>
<USER_ID_11> <USER_GENDER_M> <USER_AGE_18> <USER_OCCUPATION_15>
</USER>
<ITEM_LIST>
<L1_4> <L2_42> <L3_57> <C_0>
<L1_59> <L2_28> <L3_52> <C_0>
</ITEM_LIST>
```

每组四个 SID token 对应一部电影，按历史顺序拼接；换行仅便于展示，历史物品之间不再添加旧的 `<ITEM>` token。目标标签仍是下一部电影的 SID 加 `<EOS>`，不包含用户前缀。推荐推理时只提供用户属性和历史，目标仅用于评估。

用户字段和 SID 都作为原子 token 映射为整数 ID，不使用自然语言分词器。词表从 SID 映射及训练/验证样本的用户字段建立，随 checkpoint 保存，测试时复用。默认 `d_model=256`，全部 token 使用 T5 从零学习的共享 embedding；用户 token 也位于共享输出词表中，生成时由 SID Trie 排除。词表和参数量随数据变化，不能再按旧版155词、约740万参数固定估算。

当前实现没有新用户 ID 的未知词回退，评测面向同一用户集合的时序切分；不要将其描述为已支持任意未见用户的冷启动推荐。

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
