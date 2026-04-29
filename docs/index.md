# MLSys 菜狗自学笔记

MLSys 自学笔记，每篇尽量带可运行代码，知乎同步更新。

**[Why write this?]**

知乎/Github 好资源很多，但感觉没有特别系统性 + 满足个人逻辑闭环的 + code demo 足够的，所以想自己推一遍，每个 topic 从基础出发，补足细节，配上手写的可运行代码 / 遇到过的面试题。

计划覆盖 pre-training、post-training、inference、算子与异构计算。

qwq 其实 25 年就想写了，一直在拖，上班了继续慢慢更新中（

## 笔记列表

每篇笔记有 repo 内的 md 版本和对应的知乎 link；相关 code 有可下载的 ipynb，也有可以直接运行的 Google Colab 版本。

| Topic | 笔记 | 知乎 | code link | code简介 |
|---|------|------|-----------|----------|
| 模型基础 | [01. Transformer 详解](01_模型基础/01_01_Transformer 详解.md) | [link](https://zhuanlan.zhihu.com/p/28364382951) | [ipynb](https://github.com/Zoey-Cheng/MLSys-Learning-Notes/blob/main/code/01-mini-llama.ipynb) ｜ [colab](https://drive.google.com/file/d/1_R9oORTHsXZTkbW9OEUmRz8azSxLGtmW/view?usp=drive_link) | mini Llama |
| 训练策略 | [01. 分布式训练(0) - 背景知识(通信原语 & NCCL & 单卡计算流)](02_训练策略/02_01_分布式训练(0) - 背景知识(通信原语 & NCCL & 单卡计算流).md) | [link](https://zhuanlan.zhihu.com/p/1897578451143221835) | [ipynb](https://github.com/Zoey-Cheng/MLSys-Learning-Notes/blob/main/code/03-distributed-demo.ipynb) ｜ [colab](https://drive.google.com/file/d/15V25khFs8M8Ui3LW-_zDXUkquQytvBgq/view?usp=drive_link) | 通信原语 + DDP demo |
| 推理优化 | *coming soon* | | | |
| RL | *coming soon* | | | |
| 算子 | [01. 算子手写(1) - CUDA 入门 op](05_算子/05_01_算子手写(1) - CUDA 入门 op.md) | [link](https://zhuanlan.zhihu.com/p/1892487783110644443) | [ipynb](https://github.com/Zoey-Cheng/MLSys-Learning-Notes/blob/main/code/02-cuda-ops.ipynb) ｜ [colab](https://drive.google.com/file/d/1tcFq7B5rouZHKX239F4514f-_INscfvm/view?usp=drive_link) | 一些基础CUDA算子 |

**[About Me]**

暂时偏 DL Compiler (算子) 方向。Master NG 菜狗

- **Intern**：ByteDance CN @ AML 科学计算 → NVIDIA CN @ cuTile → NVIDIA US @ XLA
- **Fulltime**：ByteDance US @ AI Search, ML infra - 训练支持
