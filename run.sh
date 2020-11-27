clear
cd ./test
rlaunch --cpu=4 --gpu=6 --memory=32768 -- python3 inference.py --gpu_id -1 --config model1 --gpu_num 6 --config_type test
rlaunch --cpu=4 --gpu=6 --memory=32768 -- python3 inference.py --gpu_id -1 --config model2 --gpu_num 6 --config_type test
rlaunch --cpu=4 --gpu=6 --memory=32768 -- python3 inference.py --gpu_id -1 --config model3 --gpu_num 6 --config_type test
rlaunch --cpu=4 --gpu=6 --memory=32768 -- python3 inference.py --gpu_id -1 --config model4 --gpu_num 6 --config_type test
rlaunch --cpu=4 --gpu=6 --memory=32768 -- python3 inference.py --gpu_id -1 --config model5 --gpu_num 6 --config_type test
python3 getmean.py --avg_th 1.35 --vote_th 0.3 --config_name model1 model2 model3 model4 model5
#python inference.py --gpu_id -1 --config_name model2 --gpu_num 2 --config_type test
#python inference.py --gpu_id -1 --config_name model3 --gpu_num 2 --config_type test
#python inference.py --gpu_id -1 --config_name model4 --gpu_num 2 --config_type test
#python inference.py --gpu_id -1 --config_name model5 --gpu_num 2 --config_type test
#python getmean.py --avg_th 1.35 --vote_th 0.3 --config_name model1 model2 model3 model4 model5
