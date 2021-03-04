import os.path as osp
import torch
import torch.backends.cudnn as cudnn
from cmr.cmr_sg import CMR_SG
from utils.read import spiral_tramsform
from utils import utils
from options.base_options import BaseOptions
from datasets.FreiHAND.freihand import FreiHAND
from torch.utils.data import DataLoader
from run import Runner
from termcolor import cprint

if __name__ == '__main__':
    # get config
    args = BaseOptions().parse()

    # dir prepare
    args.work_dir = osp.dirname(osp.realpath(__file__))
    data_fp = osp.join(args.work_dir, 'data', args.dataset)
    args.out_dir = osp.join(args.work_dir, 'out', args.dataset, args.exp_name)
    args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')
    utils.makedirs(osp.join(args.out_dir, args.phase))
    utils.makedirs(args.out_dir)
    utils.makedirs(args.checkpoints_dir)

    # device set
    if -1 in args.device_idx or not torch.cuda.is_available():
        device = torch.device('cpu')
    elif len(args.device_idx) == 1:
        device = torch.device('cuda', args.device_idx[0])
    else:
        device = torch.device('cuda')
    torch.set_num_threads(args.n_threads)

    # deterministic
    cudnn.benchmark = True
    cudnn.deterministic = True

    template_fp = osp.join(args.work_dir, 'template', 'template.ply')
    transform_fp = osp.join(args.work_dir, 'template', 'transform.pkl')
    spiral_indices_list, down_transform_list, up_transform_list, tmp = spiral_tramsform(transform_fp, template_fp, args.ds_factors, args.seq_length, args.dilation)
    # model
    if args.model == 'cmr_sg':
        model = CMR_SG(args, spiral_indices_list, up_transform_list)
    else:
        raise Exception('Model {} not support'.format(args.model))

    # load
    if args.resume:
        if len(args.resume.split('/')) > 1:
            model_path = args.resume
        else:
            model_path = osp.join(args.checkpoints_dir, args.resume)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        cprint('Load checkpoint {}'.format(model_path), 'yellow')
    model = model.to(device)

    # run
    runner = Runner(args, model, tmp['face'], device)
    if args.phase == 'eval':
        # dataset
        eval_dataset = FreiHAND(data_fp, 'evaluation', args, tmp['face'])
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)
        runner.set_eval_loader(eval_loader)
        runner.evaluation()
    elif args.phase == 'demo':
        runner.demo()
    else:
        raise Exception('phase error')
