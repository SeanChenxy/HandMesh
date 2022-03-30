phase='eval_withgt'
exp_name='cmr_pg'
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
    --att 'yes' \
    --ds_factors 3.5 3.5 3.5 3.5 \
    --device_idx 3 \
    --resume 'cmr_pg_res18_h36m.pt'

