clear
cd ./train
rlaunch --cpu=4 --gpu=8 --memory=65536 -- python3 train.py --config model_normal --config_type train --gpu_id -1 --gpu_num 8
