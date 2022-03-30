phase='train'
exp_name='cmr_pg_h36m'
backbone='ResNet18'
dataset='Human36M'
model='cmr_pg'
python cmr/main.py \
    --phase $phase \
    --exp_name $exp_name \
    --dataset $dataset \
    --model $model \
    --backbone $backbone \
    --size 256 \
    --ds_factors 3.5 3.5 3.5 3.5 \
    --epochs 25 \
    --decay_step 20 \
    --device_idx 2