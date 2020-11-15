clear
rlaunch --cpu=4 --gpu=0 --memory=4096 -- python3 getmean.py \
--th $1 \
--ori_img_dir /data/dongchengbo/test_image \
--save_dir /data/dongchengbo/xception_0.6_th
