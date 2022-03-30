phase='demo'
exp_name='cmr_pg'
backbone='ResNet18'
dataset='FreiHAND'
model='cmr_pg'
python cmr/main.py \
    --phase $phase \
    --exp_name $exp_name \
    --dataset $dataset \
    --model $model \
    --backbone $backbone \
    --device_idx -1 \
    --resume 'cmr_pg_res18_freihand.pt'
