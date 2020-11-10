clear
GPU=$1
PRE=$2
# TIME=date +%-H_%-M
rlaunch --cpu=4 --gpu=$GPU --memory=49152 -- python3 train.py \
--prefix $PRE \
--batchSize $4 \
--gpu_id $3 \
--resume none \
--lr 0.001 \
--gpu_num $GPU \
--loss_type $5
# >> log/out_$2.log 2>&1 &