phase='train'
exp_name='cmr_pg_train'
backbone='ResNet18'
dataset='FreiHAND'
model='cmr_pg'
python main.py \
    --phase $phase \
    --exp_name $exp_name \
    --dataset $dataset \
    --model $model \
    --backbone $backbone \
    --device_idx 0 \
