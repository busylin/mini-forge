# 数据与模型许可说明

本仓库的 Apache-2.0 许可只覆盖仓库作者有权许可的自有代码，不覆盖第三方数据、模型权重、第三方库或其衍生产物。

## MovieLens 1M

MovieLens 1M 来自 [GroupLens](https://grouplens.org/datasets/movielens/1m/)。其官方 README 明确规定：

- 数据仅可在所列条件下用于研究；
- 使用数据产生论文时必须致谢/引用；
- 未经单独许可不得再分发数据；
- 未经 GroupLens 许可不得用于商业或产生收入的用途；
- 不得暗示 University of Minnesota 或 GroupLens 对本项目背书。

因此，本仓库：

- 不包含 `ml-1m.zip`、`ratings.dat`、`movies.dat` 或 `users.dat`；
- 不包含处理后的用户序列、交互表、embedding 或 MovieID-SID 映射；
- 只提供官方下载和处理代码；
- 要求使用者在下载前自行阅读并遵守[官方许可原文](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt)。

如计划商业使用本项目，请先取得 MovieLens 数据权利方所要求的许可，或替换为拥有适当商业授权的数据集。

## all-MiniLM-L6-v2

默认内容编码器为 [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)，其模型页面当前标注为 Apache-2.0。本仓库不再分发其权重，运行时由使用者从模型托管平台获取。发布模型权重或衍生产物前，仍应再次核对当时的模型卡和许可。

## Mini-T5

本项目通过 Hugging Face Transformers 的 `T5ForConditionalGeneration` 构建小型 T5，并使用动态 SID 词表从头初始化；仓库不包含 T5-small/T5-base 预训练权重。Transformers 代码库采用 Apache-2.0，但具体依赖与模型的许可仍彼此独立。

## 用户生成的产物

即使某个文件由本项目代码生成，也不代表该文件自动归属于 Apache-2.0。其可分发性可能同时受到输入数据、模型权重和其他第三方材料许可的约束。默认请不要公开：

- 基于 MovieLens 生成的处理数据；
- MovieID-SID 映射和内容 embedding；
- 训练 checkpoint；
- 包含逐用户历史或预测的文件。

本说明仅用于项目风险管理，不构成法律意见。对商业发布、数据再分发或模型权重发布存在疑问时，应咨询有资质的法律专业人士。

