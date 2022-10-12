# Deep Class Incremental Learning from Decentralized Data （IEEE TNNLS 2022）

[Arxiv](https://arxiv.org/abs/2203.05984) | [CVF] 
(https://arxiv.org/abs/2203.05984) 
###  This is the official implement of TNNLS 2022 paper "Deep Class Incremental Learning from Decentralized Data". 
## Introduction
In this paper, we focus on a new and challenging decentralized machine learning paradigm in which there are continuous inflows of data to be addressed and the data arestored in multiple repositories. We initiate the study of data decentralized class-incremental learning (DCIL) by making the following contributions. Firstly, we formulate the DCIL problem and develop theexperimental protocol. Secondly, we introduce a paradigm to create a basic decentralized counterpart of typical (centralized) class-incremental learning approaches, and as a result, establish a benchmark for the DCIL study. Thirdly, we further propose a Decentralized Composite knowledge Incremental Distillation framework (DCID) to transfer knowledge from historical models and multiple local sites to the general model continually. DCID consists of three main components namely local class-incremental learning, collaborated knowledge distillation among local models, and aggregated knowledge distillation from local models to the general one. We comprehensively investigate our DCID framework by using different implementation of the three components. Extensive experimental results demonstratethe effectiveness of our DCID framework.

![](imgs/fcil_framework.pdf)


## Code

### Install dependencies

torch >= 1.0 torchvision opencv numpy scipy, all the dependencies can be easily installed by pip or conda

This code was tested with python 3.7  

###  Train and Test

1、 Dowload Dataset ImageNet
2、 Train and Test Model 
```
sh script/sub.sh
```

### Results


## Citation
If you use this code for your research, please cite our paper:

```
@inproceedings{DCID2022,
  title={Deep Class Incremental Learning from Decentralized Data},
  author={},
  booktitle={},
  pages={},
  year={2022}
}
```
