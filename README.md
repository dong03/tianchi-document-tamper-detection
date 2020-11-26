# 不读书不配拍脑袋队

## Environment

+ Ubuntu 16.04.6 LTS
+ Python 3.6

## Requirement
+ 安装 nvidia-apex, 详见 https://github.com/NVIDIA/apex  并确保在code目录下
+ pip install requirements.txt


## Introduction
###数据处理
训练集数据均经过滑窗分patch预处理作为模型输入，预处理参数例如patch大小、滑窗步长等超参数见./code/config下的config文件。

涉及如下数据增强策略：
1) 图像压缩、图像翻转等常见数据增强策略，增强代码见./code/train/transforms.py。可在./code/config中改变aug参数选择是否采用该策略，同时可改变hard_aug参数选择是否采用高概率数据增强进行模型训练。
2) 随机拼接图片块，增强代码见./code/train/dataset.py。可在./code/config中改变random_crop参数选择是否采用该策略。

###模型结构
采用DeepLabv3_plus_res101作为backbone，可在./code/config中改变ela参数选择是否增加固定参数的SRM层对输入数据进行预先处理。

###训练策略

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