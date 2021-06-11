import torch
from collections import OrderedDict

path = 'out/Human36M/cmr_pg_h36m_resumepp_lr5/checkpoints/'
old_name = 'checkpoint_best.pt'
new_name = 'cmr_pg_res18_h36m.pt'
new_weight = OrderedDict()
checkpoint = torch.load(path+old_name, map_location='cpu')['model_state_dict']

for k, v in checkpoint.items():
    if 'backbone.' not in k and 'reduce.' not in k:
        new_weight[k] = v

torch.save(new_weight, path+new_name)
