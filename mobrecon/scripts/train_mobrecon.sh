exp_name='mrc_ds'

CUDA_VISIBLE_DEVICES=0 python -m mobrecon.main \
    --exp_name $exp_name \
    --config_file mobrecon/configs/mobrecon_ds.yml
