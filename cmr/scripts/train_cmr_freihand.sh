phase='train'
exp_name='cmr_g_freihand'
backbone='ResNet18'
dataset='FreiHAND'
model='cmr_g'
python cmr/main.py \
    --phase $phase \
    --exp_name $exp_name \
    --dataset $dataset \
    --model $model \
    --backbone $backbone \
    --device_idx 0
