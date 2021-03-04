import torch
from collections import OrderedDict

path = '../out/FreiHAND/cmr_sg/checkpoints/'
old_name = 'cmr_sg.pt'
new_name = 'cmr_sg_res18_freihand.pt'
new_weight = OrderedDict()
checkpoint = torch.load(path+old_name, map_location='cpu')['model_state_dict']

for k, v in checkpoint.items():
    if 'backbone.' not in k:
        new_weight[k] = v

torch.save(new_weight, path+new_name)