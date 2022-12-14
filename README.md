# Deep Class Incremental Learning from Decentralized Data （IEEE TNNLS 2022）

[Arxiv](https://arxiv.org/abs/2203.05984) | [Official Version] 

###  This is the official implement of TNNLS 2022 paper "Deep Class Incremental Learning from Decentralized Data". 

## Introduction
In this paper, we focus on a new and challenging decentralized machine learning paradigm in which there are continuous inflows of data to be addressed and the data arestored in multiple repositories. We initiate the study of data decentralized class-incremental learning (DCIL) by making the following contributions. Firstly, we formulate the DCIL problem and develop theexperimental protocol. Secondly, we introduce a paradigm to create a basic decentralized counterpart of typical (centralized) class-incremental learning approaches, and as a result, establish a benchmark for the DCIL study. Thirdly, we further propose a Decentralized Composite knowledge Incremental Distillation framework (DCID) to transfer knowledge from historical models and multiple local sites to the general model continually. DCID consists of three main components namely local class-incremental learning, collaborated knowledge distillation among local models, and aggregated knowledge distillation from local models to the general one. We comprehensively investigate our DCID framework by using different implementation of the three components. Extensive experimental results demonstratethe effectiveness of our DCID framework.

![](imgs/fcil_framework.png)

## Code

### Install dependencies

torch >= 1.0 torchvision opencv numpy scipy, all the dependencies can be easily installed by pip or conda.

This code was tested with python 3.7.

We refer to LUCIR (https://github.com/hshustc/CVPR19_Incremental_Learning) to setup the experimental environment.

###  Train and Test

1、 Dowload Dataset ImageNet and Selected 100 Classes Randomlly 

2、 Train and Test Model 
```
sh script/sub.sh  # our DCID method

sh script/baseline.sh   # baseline method
```
### Results
![](imgs/sub.png)

## Citation
If you use this code for your research, please cite our paper:

```
@ARTICLE{zhang2022dcid,
  author={Zhang, Xiaohan and Dong, Songlin and Chen, Jinjie and Tian, Qi and Gong, Yihong and Hong, Xiaopeng},
  journal={IEEE Transactions on Neural Networks and Learning Systems},   
  title={Deep Class Incremental Learning from Decentralized Data},   
  year={2022},  
  volume={},
  number={},
  pages={},  
  doi={10.1109/TNNLS.2022.3214573}
  }

```
