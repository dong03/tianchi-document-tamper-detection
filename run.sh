inference_config1=sth
inference_config2=sth
python3 inference.py --gpu_id 0 --config_name $inference_config1
python3 inference.py --gpu_id 0 --config_name $inference_config1
python3 inference.py --gpu_id 0 --config_name $inference_config1
python3 inference.py --gpu_id 0 --config_name $inference_config1
python3 inference.py --gpu_id 0 --config_name $inference_config1
python3 getmean.py --avg_th 1.35 --vote_th 0.3 \
--model_name $inference_config1 \
$inference_config1 \
$inference_config1 \
$inference_config1 \
$inference_config1
