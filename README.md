# Kaggle竞赛：Stanford RNA 3D Folding

#### *author - Chengzhi Jiang*

**持续更新中 | 最后更新：2025-04-23**  
[![Kaggle Profile](https://img.shields.io/badge/Kaggle-Profile-blue?logo=kaggle)](https://www.kaggle.com/competitions/stanford-rna-3d-folding)

## 项目概述
本项目是作者首次参与的Kaggle竞赛代码仓库，记录从基线模型到优化方案的完整迭代过程。通过本仓库您可以：
- 🚀 获取作者模型的代码进行复现  
- 📊 查看我的模型优化思路,并基于此改进  
- 🤝 ​**欢迎通过我的邮箱提交您的批评指正以及优化建议 2376305851@qq.com**  

### 比赛背景
> RNA is vital to life’s most essential processes, but despite its significance, predicting its 3D structure is still difficult. Deep learning breakthroughs like AlphaFold have transformed protein structure prediction, but progress with RNA has been much slower due to limited data and evaluation methods.
This competition builds on recent advances, like the deep learning foundation model RibonanzaNet, which emerged from a prior Kaggle competition. Now, you’ll take on the next challenge—predicting RNA’s full 3D structure.
Your work could push RNA-based medicine forward, making treatments like cancer immunotherapies and CRISPR gene editing more accessible and effective. More fundamentally, your work may be the key step in illuminating the folds and functions of natural RNA molecules, which have been called the 'dark matter of biology'.
This competition is made possible through a worldwide collaborative effort including the organizers, experimental RNA structural biologists, and predictors of the CASP16 and RNA-Puzzles competitions; Howard Hughes Medical Institute; the Institute of Protein Design; and Stanford University School of Medicine.  
> ​**赛事类型**：Featured Competition    
> ​**技术方向**：深度学习     
> ​**评估指标**：TM-score  

### 我的思路
> 在Ribonanza是一个优秀的基于rna序列对齐进行二级结构,三级结构预测的模型.基于最初的Ribonanza模型，我对此进行了改进，当然在比赛中，已经有Ribonanza的升级版了，但是我还是基于最初的Ribonanza模型进行优化，最终分数当然肯定不如大佬的，但是相信自己也能有所提升   
> Mamba在近几年异常火热,尤其是最近Mamba和Transformer的Hybrid版本层出不穷,我采用暴力优化方法，以计算复杂度、计算时间换准确度的方式,添加SelectiveSSM模块,并且采用可分离卷积替换掉原架构的普通1维卷积,因为可分离卷积的不同ksize能够更好地捕获RNA的各个特征,同时采用非传统卷积 - 因果卷积+门控机制的融合,在生物学上非常合理.

#### 项目架构

├─ BadVerion - 尝试通过CNN捕获特征,采用了Kaggle上基于CNN的 baseline，并基于此进行优化，效果并不理想分数极低       
│           
├─ Pro    
│    ├─ Baseline - 我参考并复现的 RibonanzaNet 的核心代码   
│    │       
│    ├─ MyOptimization - 我对于 RibonanzaNet 的改造     
│    │    
│    └─ README.md - 关于我的魔改模型的 README  
│           
└─ README - 有关这次比赛的信息 
        

#### *This Repository is for me(a s1mple student) to record the game,tks for ur comments*
