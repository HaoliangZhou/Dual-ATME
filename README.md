# Dual-ATME


> **Dual-ATME: Dual-branch Attention Network for Micro-Expression Recognition** ([Paper](https://doi.org/10.3390/e25030460))<br>
> Haoliang Zhou, Shucheng Huang, Jingting Li and SuJing Wang<br> 

> **Abstract**: <br>
> Micro-expression recognition (MER) is challenging due to the difficulty of capturing the instantaneous and subtle motion changes of micro-expressions (MEs). Early works based on hand-crafted features extracted from prior knowledge showed some promising results, but have recently been replaced by deep learning methods based on attention mechanism. However, with limited ME sample size, features extracted by these methods lack discriminative ME representations, in yet-to-be improved MER performance. 
> This paper proposes the Dual-branch Attention Network (Dual-ATME) for MER to address the problem of ineffective single-scale features representing MEs. Specifically, Dual-ATME consists of two components: Hand-crafted Attention Region Selection (HARS) and Automated Attention Region Selection (AARS). HARS uses prior knowledge to manually extract features from regions of interest (ROIs). Meanwhile, AARS is based on attention mechanisms and extracts hidden information from data automatically. Finally, through similarity comparison and feature fusion, the dual-scale features could be used to learn ME representations effectively. 
> Experiments on spontaneous ME >datasets (including CASME II, SAMM, SMIC) and their composite dataset MEGC2019-CD show that Dual-ATME achieves better or more competitive performance than the state-of->the-art MER methods.


> **Citation**: <br>
> If you find this repo useful for your research, please consider citing the paper
```
@Article{zhou2023dualatme,
AUTHOR = {Zhou, Haoliang and Huang, Shucheng and Li, Jingting and Wang, Su-Jing},
TITLE = {Dual-ATME: Dual-Branch Attention Network for Micro-Expression Recognition},
JOURNAL = {Entropy},
VOLUME = {25},
YEAR = {2023},
URL = {https://www.mdpi.com/1099-4300/25/3/460},
ISSN = {1099-4300},
DOI = {10.3390/e25030460}
}
```
