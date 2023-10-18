# Dual-ATME


### Dual-ATME: Dual-branch Attention Network for Micro-Expression Recognition [![paper](https://img.shields.io/badge/Paper-87CEEB)](https://doi.org/10.3390/e25030460) <br>
*Haoliang Zhou, Shucheng Huang, Jingting Li, SuJing Wang*<br> 

##

### Abstract <br>
Micro-expression recognition (MER) is challenging due to the difficulty of capturing the instantaneous and subtle motion changes of micro-expressions (MEs). Early works based on hand-crafted features extracted from prior knowledge showed some promising results, but have recently been replaced by deep learning methods based on attention mechanism. However, with limited ME sample size, features extracted by these methods lack discriminative ME representations, in yet-to-be improved MER performance. 
This paper proposes the Dual-branch Attention Network (Dual-ATME) for MER to address the problem of ineffective single-scale features representing MEs. Specifically, Dual-ATME consists of two components: Hand-crafted Attention Region Selection (HARS) and Automated Attention Region Selection (AARS). HARS uses prior knowledge to manually extract features from regions of interest (ROIs). Meanwhile, AARS is based on attention mechanisms and extracts hidden information from data automatically. Finally, through similarity comparison and feature fusion, the dual-scale features could be used to learn ME representations effectively. 
Experiments on spontaneous ME >datasets (including CASME II, SAMM, SMIC) and their composite dataset MEGC2019-CD show that Dual-ATME achieves better or more competitive performance than the state-of->the-art MER methods.

<p align="center">
<img src="https://github.com/HaoliangZhou/Dual-ATME/blob/master/fig.png" width=100% height=100% 
class="center">
</p>

### Data preparation
Following [RCN](https://github.com/xiazhaoqiang/ParameterFreeRCNs-MicroExpressionRec/blob/main/PrepareData_LOSO_CD.py), the data lists are reorganized as follow:

```
data/
├─ MEGC2019/
│  ├─ v_cde_flow/
│  │  ├─ 006_test.txt
│  │  ├─ 006_train.txt
│  │  ├─ 007_test.txt
│  │  ├─ ...
│  │  ├─ sub26_train.txt
│  │  ├─ subName.txt
```
1. There are 3 columns in each txt file: 
```
/home/user/data/samm/flow/006_006_1_2_006_05588-006_05562_flow.png 0 1
```
In this example, the first column is the path of the optical flow image for a particular ME sample, the second column is the label (0-2 for three emotions), and the third column is the database type (1-3 for three databases).

2. There are 68 raws in _subName.txt_: 
```
006
...
037
s01
...
s20
sub01
...
sub26
```
Represents ME samples divided by MEGC2019, as described in [here](https://facial-micro-expressiongc.github.io/MEGC2019/) ahd [here](https://facial-micro-expressiongc.github.io/MEGC2019/images/MEGC2019%20Recognition%20Challenge.pdf).


### Citation <br>
If you find this repo useful for your research, please consider citing the paper
```
@Article{zhou2023dualatme,
AUTHOR = {Zhou, Haoliang and Huang, Shucheng and Li, Jingting and Wang, Su-Jing},
TITLE = {Dual-ATME: Dual-Branch Attention Network for Micro-Expression Recognition},
JOURNAL = {Entropy},
VOLUME = {25},
YEAR = {2023},
ISSN = {1099-4300},
DOI = {10.3390/e25030460}
}
```
