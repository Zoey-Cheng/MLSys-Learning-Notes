# MLSys 菜狗自学笔记

MLSys 自学笔记，知乎同步更新。

**[Why write this?]**

知乎好文很多，感觉没有特别系统性 + 从头形成逻辑闭环的，……所以想自己推一遍，每个 topic 从基础出发，补足细节，配上手写的可运行代码。

计划覆盖 pre-training、post-training、inference、算子与异构计算，目前还在持续更新中。

## 笔记列表

每篇笔记有 repo 内的 md 版本和对应的知乎 link；相关 code 有可下载的 ipynb，也有可以直接运行的 Google Colab 版本。

| # | 笔记 | 知乎 | code link | code简介 |
|---|------|------|-----------|----------|
| 01 | [Transformer 详解](notes/01_Transformer 详解.md) | [link](https://zhuanlan.zhihu.com/p/28364382951) | [ipynb](https://github.com/Zoey-Cheng/MLSys-Learning-Notes/blob/main/code/01-mini-llama.ipynb) ｜ [colab](https://drive.google.com/file/d/1_R9oORTHsXZTkbW9OEUmRz8azSxLGtmW/view?usp=drive_link) | mini Llama |
| 02 | [算子手写(1) - CUDA 入门 op](notes/02_算子手写(1) - CUDA 入门 op.md) | [link](https://zhuanlan.zhihu.com/p/1892487783110644443) | [ipynb](https://github.com/Zoey-Cheng/MLSys-Learning-Notes/blob/main/code/02-cuda-ops.ipynb) ｜ [colab](https://drive.google.com/file/d/1tcFq7B5rouZHKX239F4514f-_INscfvm/view?usp=drive_link) | 一些基础CUDA算子 |
| 03 | [分布式训练(0) - 背景知识(通信原语 & NCCL & 单卡计算流)](notes/03_分布式训练(0) - 背景知识(通信原语 & NCCL & 单卡计算流).md) | [link](https://zhuanlan.zhihu.com/p/1897578451143221835) | [ipynb](https://github.com/Zoey-Cheng/MLSys-Learning-Notes/blob/main/code/03-distributed-demo.ipynb) ｜ [colab](https://drive.google.com/file/d/15V25khFs8M8Ui3LW-_zDXUkquQytvBgq/view?usp=drive_link) | 通信原语 + DDP demo |

**[个人 Background]**

偏 DL Compiler (算子) 方向。Master NG 菜狗

- **Intern**：ByteDance CN @ AML 科学计算 → NVIDIA CN @ cuTile → NVIDIA US @ XLA
- **Fulltime**：ByteDance US @ AI Search, ML infra - 训练支持
