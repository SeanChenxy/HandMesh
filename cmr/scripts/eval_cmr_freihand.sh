phase='eval'
exp_name='cmr_sg'
backbone='ResNet18'
dataset='FreiHAND'
model='cmr_sg'
python cmr/main.py \
    --phase $phase \
    --exp_name $exp_name \
    --dataset $dataset \
    --model $model \
    --backbone $backbone \
    --device_idx -1 \
    --resume 'cmr_sg_res18_freihand.pt' \
