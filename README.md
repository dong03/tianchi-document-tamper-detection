# 不读书不配拍脑袋队

## Environment

+ Ubuntu 16.04.6 LTS
+ Python 3.6
+ cuda10.1+cudnn7.6.3

## Requirement
+ 安装 nvidia-apex, 详见 https://github.com/NVIDIA/apex  并确保在code目录下
+ pip install requirements.txt


## Introduction
### 数据处理

### 模型结构

当作分割任务
模型主体采用deeplabv3-plus, backbone:resnet101 (详见https://github.com/MLearing/Pytorch-DeepLab-v3-plus)

考虑到篡改部分噪声可能与原图不同,在模型中引入SRM噪声核，forward时生成3通道噪声图，并与原始图片拼接为6通道，作为输入(可选)
![img-srm](https://images.gitee.com/uploads/images/2020/1126/183406_22693a27_5469175.png "img_srm.png")
### 训练策略

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