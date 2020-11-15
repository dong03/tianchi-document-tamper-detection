clear
rlaunch --cpu=8 --gpu=1 --memory=4096 -- python3 inference.py \
--gpu_id 0 \
--resume /data/dongchengbo/tianchi_checkpoints/deeplab_xception_256_4loss_stride8/model_best.pt \
--remove 1 \
--prefix $1

# cd /data/dongchengbo/VisualSearch/tianchi_s2/s2_data/data
# zip images.zip ./images/*
# mv ./images.zip $HOME/code/ClassNSeg/
# cd $HOME/code/ClassNSeg
#'/data/dongchengbo/code/ClassNSeg/checkpoints/full/b3_aug_new/model_18.pt'
#'/data/dongchengbo/code/ClassNSeg/checkpoints/full/half_dce/model_6.pt'

#'/data/dongchengbo/code/ClassNSeg/checkpoints/full/b0_128patch_noaug/model_best.pt'

#/data/dongchengbo/tianchi_checkpoints/b0_aug_256_s/model_best.pt
#/data/dongchengbo/tianchi_checkpoints/b0_aug_256_s/model_0.pt

