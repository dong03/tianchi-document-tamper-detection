clear
cd ./train
python3 train.py --config model1 --config_type train --gpu_id -1 --gpu_num 2
python3 train.py --config model2 --config_type train --gpu_id -1 --gpu_num 2
python3 train.py --config model3 --config_type train --gpu_id -1 --gpu_num 2
python3 train.py --config model4 --config_type train --gpu_id -1 --gpu_num 2
python3 train.py --config model5 --config_type train --gpu_id -1 --gpu_num 2