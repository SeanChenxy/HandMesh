from torch.nn.parameter import Parameter


def costom_load_state_dict(curr_model, prev_model, start=0, end=None, to_cpu=True):
    own_state = curr_model.state_dict()
    prev_state = prev_model
    #import ipdb; ipdb.set_trace()
    for name, param in list(prev_state.items())[start:end]:
        # print(name)
        if name not in own_state and name.startswith('module'):
            name = name[7:]
        if name not in own_state:
            print('Unexpected key "{}" in state_dict'.format(name))
            continue
            #raise KeyError('Unexpected key "{}" in state_dict'.format(name))
        if isinstance(param, Parameter):
            param = param.data
        try:
            if to_cpu:
                own_state[name].copy_(param.cpu())
            else:
                own_state[name].copy_(param)
        except:
            print('Note: the parameter {} is inconsistent!'.format(name))
            continue
