# 不读书不配拍脑袋队

## Environment

+ Ubuntu 16.04.6 LTS
+ Python 3.6
+ cuda10.1+cudnn7.6.3

## Requirement
+ 安装 [nvidia-apex](https://github.com/NVIDIA/apex)
, 并确保在code目录下
+ pip install requirements.txt


## Introduction
###数据处理
鉴于目标篡改区域于全图相比面积可能较小, 训练集数据均经过滑窗分patch预处理作为模型输入, 预处理参数例如patch大小、滑窗步长等超参数见./code/config下的config文件。

训练策略涉及如下数据增强策略：
1) 图像压缩、图像翻转等常见数据增强策略, 增强代码见./code/train/transforms.py。可在./code/config中改变aug参数选择是否采用该策略, 同时可改变hard_aug参数选择是否采用高概率数据增强进行模型训练。
2) 随机拼接图片块, 增强代码见./code/train/dataset.py。可在./code/config中改变random_crop参数选择是否采用该策略。

###模型结构
模型主体采用[deeplabv3-plus](https://github.com/MLearing/Pytorch-DeepLab-v3-plus), backbone:resnet101,模型输出为1通道-同尺寸的map,数值越接近1即该像素被篡改概率越大

考虑到篡改部分噪声可能与原图不同,在模型中引入SRM噪声核, forward时生成3通道噪声图, 并与原始图片拼接为6通道作为输入。可在./code/config中改变ela参数选择是否采用该策略。
![img-srm](https://images.gitee.com/uploads/images/2020/1126/183406_22693a27_5469175.png "img_srm.png")

### 训练策略
1. 采用4项损失函数
    + MSELoss
    + DICELoss
    + BCELoss
    + FocalLoss

    其中,FocalLoss赋予真实图片对应的mask(全黑)的损失权重高于篡改图片的损失权重, 并使用Auto-Weighted-Loss策略设置损失函数的权重
2. 采用学习率衰减的Adam优化器
3. 使用nvidia-apex半精度训练
### 推理策略
1. 分数映射 
    + 将模型输出的0-1连续值map依据近似sigmoid的策略进行映射, 使分数分布更集中于高分段/低分段, 方差更小
2. 二值化
    + 依据预测值方差动态确定二值化阈值, 见./code/test/inference.py
3. 模型集成
    + 选择不同参属下训练、推理的结果进行后融合, 在二值化后使用Voting策略决定最终结果. 见./code/test/getmean.py

## Usage
### Test
```
cd ./code
bash run.sh
```
run.sh中默认参数是使用两块编号分别是0,1的卡推理。
gpu_id和gpu_num的使用要求为
```
--gpu_id 0  --gpu_num 1 # 仅使用一张卡,适用于单卡/多卡仅使用编号为0的卡
--gpu_id 1  --gpu_num 1 # 仅使用一张卡,适用于多卡使用编号为1
--gpu_id -1 --gpu_num 2 # 使用两张卡,gpu_id必须为-1,且多卡编号必须为从0开始的连续值
```

例如,只使用一块卡,请修改为
```
clear
cd ./test
python inference.py --gpu_id 0 --config_name model1 --gpu_num 1
python inference.py --gpu_id 0 --config_name model2 --gpu_num 1
python inference.py --gpu_id 0 --config_name model3 --gpu_num 1
python inference.py --gpu_id 0 --config_name model4 --gpu_num 1
python inference.py --gpu_id 0 --config_name model5 --gpu_num 1
python getmean.py --avg_th 1.35 --vote_th 0.3 --config_name model1 model2 model3 model4 model5
```


### Train
```
cd ./code
bash train.sh
```
其中,gpu_id和gpu_num的要求与Test一致

其余参数请自行在./code/config的config文件中修改