clear
cd .\test
python3 inference.py --gpu_id 0 --config_name model1 --gpu_num 2
python3 inference.py --gpu_id 0 --config_name model2 --gpu_num 2
python3 inference.py --gpu_id 0 --config_name model3 --gpu_num 2
python3 inference.py --gpu_id 0 --config_name model4 --gpu_num 2
python3 inference.py --gpu_id 0 --config_name model5 --gpu_num 2
python3 getmean.py --avg_th 1.35 --vote_th 0.3 --model_name model1 model2 model3 model4 model5
