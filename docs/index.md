# MLSys 菜狗自学笔记

MLSys 自学笔记，每篇尽量带可运行代码，知乎同步更新。

**[About Planning]**

计划覆盖 pre-training、post-training、inference、算子与异构计算。大致目标：

- 基础模型：了解结构，能手写简化版模型代表
- 训练：前后向计算流程，DP/EP/SP/CP/TP/PP 并行 + 通信量分析
- 推理：prefill / decode 分离合并，以及 KV Cache 各类相关优化
- 算子：基础 CUDA 算子，FA 1/2，进阶 - 架构相关优化、MoE 相关算子
- RL：……还没想好，DPO/PPO/GRPO，sync / async / multi turn 之类的
- 面试手写题快查 (模型 / 推理 / 算子)

qwq 其实 25 年就想写了，一直在拖，上班了继续慢慢更新中（

## 笔记列表

每篇笔记有 repo 内的 md 版本和对应的知乎 link；相关 code 有可下载的 ipynb，也有可以直接运行的 Google Colab 版本。

| Topic | 笔记 | 知乎 | code link | code简介 |
|---|------|------|-----------|----------|
| 模型基础 | [01-Transformer 详解 (Llama为例)](01_模型基础/01_01_Transformer.md) | [link](https://zhuanlan.zhihu.com/p/28364382951) | [ipynb](https://github.com/Zoey-Cheng/MLSys-Learning-Notes/blob/main/code/01-mini-llama.ipynb) ｜ [colab](https://drive.google.com/file/d/1_R9oORTHsXZTkbW9OEUmRz8azSxLGtmW/view?usp=drive_link) | mini Llama |
|  | [02-MoE 详解 (DeepSeek V1 / Qwen3为例)](01_模型基础/01_02_MoE.md) | [link](https://zhuanlan.zhihu.com/p/2039331127815557840) | [ipynb](https://github.com/Zoey-Cheng/MLSys-Learning-Notes/blob/main/code/06_mini_moe_qwen3.ipynb) ｜ [colab](https://drive.google.com/file/d/19DwqNBulYpXuyedOMbglgauXsB183Oil/view?usp=sharing) | mini MoE Qwen3 |
|  | 03-VLM (Qwen3-VL 为例) *WIP* | | | |
|  | [04-Diffusion(上) - VAE 到 DDPM / DDIM](01_模型基础/01_04_Diffusion基础.md) | [link](https://zhuanlan.zhihu.com/p/2051913285751005547) | [ipynb](https://github.com/Zoey-Cheng/MLSys-Learning-Notes/blob/main/code/07_basic_ddim.ipynb) ｜ [colab](https://drive.google.com/file/d/1R8kfN8Qv2SQFy7XefKuEW2lebmiElTbP/view?usp=sharing) | Basic DDIM |
| 训练策略 | [01-分布式训练(0) - 背景知识(通信 & 单卡计算流)](02_训练策略/02_01_分布式训练基础.md) | [link](https://zhuanlan.zhihu.com/p/1897578451143221835) | [ipynb](https://github.com/Zoey-Cheng/MLSys-Learning-Notes/blob/main/code/03-distributed-demo.ipynb) ｜ [colab](https://drive.google.com/file/d/15V25khFs8M8Ui3LW-_zDXUkquQytvBgq/view?usp=drive_link) | 通信原语 + DDP demo |
|  | 02-数据并行 DP *WIP* | | | |
| 训练方法 | [01-预训练 Pretrain](03_训练方法/03_01_Pretrain.md) | [link](https://zhuanlan.zhihu.com/p/2033923074630870192) | [ipynb](https://github.com/Zoey-Cheng/MLSys-Learning-Notes/blob/main/code/04_mini_pretrain.ipynb) ｜ [colab](https://drive.google.com/file/d/1VUB1WrZx9KkHBrY9N8E3-Dfmql7aHnjw/view?usp=sharing) | mini pretrain |
|  | [02-监督微调 SFT](03_训练方法/03_02_SFT.md) | [link](https://zhuanlan.zhihu.com/p/2034286003767218701) | [ipynb](https://github.com/Zoey-Cheng/MLSys-Learning-Notes/blob/main/code/05_mini_lora_sft.ipynb) ｜ [colab](https://drive.google.com/file/d/1NrDWiGrWPoRrk2yszFXIkDe-DeyKW7B0/view?usp=drive_link) | mini lora SFT |
| RL | *coming soon* | | | |
| 推理优化 | 01-推理基础 *WIP* | | | |
| 算子 | [01-算子手写(1) - CUDA 入门 op](06_算子/05_01_CUDA入门.md) | [link](https://zhuanlan.zhihu.com/p/1892487783110644443) | [ipynb](https://github.com/Zoey-Cheng/MLSys-Learning-Notes/blob/main/code/02-cuda-ops.ipynb) ｜ [colab](https://drive.google.com/file/d/1tcFq7B5rouZHKX239F4514f-_INscfvm/view?usp=drive_link) | 一些基础CUDA算子 |
|  | 02-Flash Attention (上) - FA1/2 *WIP* | | | |

**[Why write this?]**

知乎/Github 好资源很多，但感觉没有特别系统性 + 满足个人逻辑闭环的 + code demo 足够的，所以想自己推一遍，每个 topic 从基础出发，补足细节，配上手写的可运行代码 / 遇到过的面试题。

**[About Me]** 26 Master NG 菜狗

- **Intern**：ByteDance CN @ AML 科学计算 → NVIDIA CN @ cuTile → NVIDIA US @ XLA
- **Fulltime**：ByteDance US @ AI Search, ML infra - 训练支持

## 更新历史

- **06/20/2026**: 模型基础 → 04-Diffusion(上) [[link]](01_模型基础/01_04_Diffusion基础.md)，网页版和知乎同步发布
- **05/16/2026**: 模型基础 → 02-MoE详解 [[link]](01_模型基础/01_02_MoE.md)，网页版和知乎同步发布
- **05/03/2026**: 训练方法 → 02-SFT [[link]](03_训练方法/03_02_SFT.md)，网页版和知乎同步发布
- **05/02/2026**: 训练方法 → 01-Pretrain [[link]](https://zhuanlan.zhihu.com/p/2033923074630870192)，网页版和知乎同步发布
- **04/25/2026**: 新开 GitHub repo，帖子迁移到网页版了！🎉🎉
- **04/20/2025**: 训练策略 → 01-分布式训练(0)-背景知识(通信 & 单卡计算流) [[link]](02_训练策略/02_01_分布式训练基础.md)，发布在知乎
- **04/06/2025**: 算子 → 01-常见CUDA手写实现 [[link]](06_算子/05_01_CUDA入门.md)，发布在知乎
- **03/06/2025**: 模型结构 → 01-Transformer详解 [[link]](01_模型基础/01_01_Transformer.md)，发布在知乎

## 访客地图

<div id="mmv-holder" style="width:100%;max-width:480px;margin:0 auto;text-align:center"></div>
