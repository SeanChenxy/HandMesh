phase='demo'
exp_name='cmr_sp'
backbone='ResNet50'
dataset='FreiHAND'
model='cmr_sp'
python main.py \
    --phase $phase \
    --exp_name $exp_name \
    --dataset $dataset \
    --model $model \
    --backbone $backbone \
    --device_idx -1 \
    --resume 'cmr_sp_res50_freihand.pt'
