clear
cd .\train
python3 train.py --gpu_id -1 --gpu_num 2 --config model1
python3 train.py --gpu_id -1 --gpu_num 2 --config model2
python3 train.py --gpu_id -1 --gpu_num 2 --config model3
python3 train.py --gpu_id -1 --gpu_num 2 --config model4
python3 train.py --gpu_id -1 --gpu_num 2 --config model5