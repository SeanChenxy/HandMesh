phase='demo'
exp_name='cmr_g'
backbone='ResNet18'
dataset='FreiHAND'
model='cmr_g'
python main.py \
    --phase $phase \
    --exp_name $exp_name \
    --dataset $dataset \
    --model $model \
    --backbone $backbone \
    --device_idx -1 \
    --resume 'cmr_g_res18_freihand.pt'
