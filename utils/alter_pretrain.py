import torch
from collections import OrderedDict

path = '../out/FreiHAND/cmr_sp/checkpoints/'
old_name = 'checkpoint.pt'
new_name = 'cmr_sp_res50_freihand.pt'
new_weight = OrderedDict()
checkpoint = torch.load(path+old_name, map_location='cpu')['model_state_dict']

for k, v in checkpoint.items():
    if 'backbone.' not in k:
        new_weight[k] = v

torch.save(new_weight, path+new_name)