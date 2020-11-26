# 不读书不配拍脑袋队

## Environment

+ Ubuntu 16.04.6 LTS
+ Python 3.6

## Requirement
+ 安装 nvidia-apex, 详见 https://github.com/NVIDIA/apex  并确保在code目录下
+ pip install requirements.txt

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
Change following in run_train.sh
```
data_path, run_id, config_name, train_collection, val_collection, test_collection
```
Then run 
```bash 
bash run_train.sh gpu_id additional_description
e.g. bash run_train.sh 0 Vfinetune
```
### Prediction
for img: 

```bash
bash run_pred_img.sh gpu_id if_submision
e.g. bash run_pred_img.sh 0 1 
```
for video:
```bash
bash run_pred_video.sh 0
```
To use pretrained models, get them from 

链接：https://pan.baidu.com/s/1fepyjxcCn6zyorAlQyvTQg 提取码：i1v9

and change model path in run_pred_xx.sh
### Visualization (via gradcam)  
```bash
bash run_gradcam.sh gpu_id
```