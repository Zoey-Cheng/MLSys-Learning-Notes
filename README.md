# MLSys 菜狗自学笔记

MLSys 自学笔记，每篇尽量带可运行代码，知乎同步更新。

📖 **网页版**：<https://zoey-cheng.github.io/MLSys-Learning-Notes/>

**[Why write this?]**

知乎/Github 好资源很多，但感觉没有特别系统性 + 满足个人逻辑闭环的 + code demo 足够的，所以想自己推一遍，每个 topic 从基础出发，补足细节，配上手写的可运行代码 / 遇到过的面试题。

计划覆盖 pre-training、post-training、inference、算子与异构计算。

qwq 其实 25 年就想写了，一直在拖，上班了继续慢慢更新中（

**[个人Background]**

暂时偏 DL Compiler (算子) 方向。Master NG 菜狗
- **Intern**：ByteDance CN @ AML 科学计算 → NVIDIA CN @ cuTile → NVIDIA US @ XLA
- **Fulltime**：ByteDance US @ AI Search, ML infra - 训练支持

## 笔记列表

每篇笔记有 repo 内的 md 版本和对应的知乎 link；相关 code 有可下载的 ipynb，也有可以直接运行的 Google Colab 版本。

| # | 笔记 | 知乎 | code link | code简介 |
|---|------|------|-----------|----------|
| 01 | [Transformer 详解](notes/01_Transformer%20详解.md) | [link](https://zhuanlan.zhihu.com/p/28364382951) | [ipynb](code/01-mini-llama.ipynb) ｜ [colab](https://drive.google.com/file/d/1_R9oORTHsXZTkbW9OEUmRz8azSxLGtmW/view?usp=drive_link) | mini Llama |
| 02 | [算子手写(1) - CUDA 入门 op](notes/02_算子手写(1)%20-%20CUDA%20入门%20op.md) | [link](https://zhuanlan.zhihu.com/p/1892487783110644443) | [ipynb](code/02-cuda-ops.ipynb) ｜ [colab](https://drive.google.com/file/d/1tcFq7B5rouZHKX239F4514f-_INscfvm/view?usp=drive_link) | 一些基础CUDA算子 |
| 03 | [分布式训练(0) - 背景知识(通信原语 & NCCL & 单卡计算流)](notes/03_分布式训练(0)%20-%20背景知识(通信原语%20&%20NCCL%20&%20单卡计算流).md) | [link](https://zhuanlan.zhihu.com/p/1897578451143221835) | [ipynb](code/03-distributed-demo.ipynb) ｜ [colab](https://drive.google.com/file/d/15V25khFs8M8Ui3LW-_zDXUkquQytvBgq/view?usp=drive_link) | 通信原语 + DDP demo |

## 结构

```
├── notes/    # Markdown 笔记
├── code/     # 配套手写代码
└── assets/   # 笔记里用到的图片
```


