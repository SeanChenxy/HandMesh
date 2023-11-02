import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os.path as osp
import torch
import torch.backends.cudnn as cudnn
from cmr.models.cmr_sg import CMR_SG
from cmr.models.cmr_pg import CMR_PG
from cmr.models.cmr_g import CMR_G
from cmr.models.mobrecon_densestack import MobRecon
from utils.read import spiral_tramsform
from utils import utils, writer
from options.base_options import BaseOptions
from cmr.datasets.FreiHAND.freihand import FreiHAND
from cmr.datasets.Human36M.human36m import Human36M
from torch.utils.data import DataLoader
from runer import Runner
from termcolor import cprint
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    # get config
    args = BaseOptions().parse()

    # dir prepare
    args.work_dir = osp.dirname(osp.realpath(__file__))
    data_fp = osp.join(args.work_dir, '../data', args.dataset)
    args.out_dir = osp.join(args.work_dir, 'out', args.dataset, args.exp_name)
    args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')
    if args.phase in ['eval', 'demo']:
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

    if args.dataset=='Human36M':
        template_fp = osp.join(args.work_dir, '../template/template_body.ply')
        transform_fp = osp.join(args.work_dir, '../template/transform_body.pkl')
    else:
        template_fp = osp.join(args.work_dir, '../template/template.ply')
        transform_fp = osp.join(args.work_dir, '../template/transform.pkl')
    spiral_indices_list, down_transform_list, up_transform_list, tmp = spiral_tramsform(transform_fp, template_fp, args.ds_factors, args.seq_length, args.dilation)

    # model
    if args.model == 'cmr_sg':
        model = CMR_SG(args, spiral_indices_list, up_transform_list)
    elif args.model == 'cmr_pg':
        model = CMR_PG(args, spiral_indices_list, up_transform_list)
    elif args.model == 'cmr_g':
        model = CMR_G(args, spiral_indices_list, up_transform_list)
    elif args.model == 'mobrecon':
        for i in range(len(up_transform_list)):
            up_transform_list[i] = (*up_transform_list[i]._indices(), up_transform_list[i]._values())
        model = MobRecon(args, spiral_indices_list, up_transform_list)
    else:
        raise Exception('Model {} not support'.format(args.model))

    # load
    epoch = 0
    if args.resume:
        if len(args.resume.split('/')) > 1:
            model_path = args.resume
        else:
            model_path = osp.join(args.checkpoints_dir, args.resume)
        checkpoint = torch.load(model_path, map_location='cpu')
        if checkpoint.get('model_state_dict', None) is not None:
            checkpoint = checkpoint['model_state_dict']
        model.load_state_dict(checkpoint)
        epoch = checkpoint.get('epoch', -1) + 1
        cprint('Load checkpoint {}'.format(model_path), 'yellow')
    model = model.to(device)

    # run
    runner = Runner(args, model, tmp['face'], device)
    if args.phase == 'train':
        # log
        writer = writer.Writer(args)
        writer.print_str(args)
        # dataset
        if args.dataset=='FreiHAND':
            eval_dataset = FreiHAND(data_fp, 'evaluation', args, tmp['face'])
            eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)
            train_dataset = FreiHAND(data_fp, 'training', args, tmp['face'], writer=writer, down_sample_list=down_transform_list, ms=args.ms_mesh)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=8, drop_last=True)
        elif args.dataset=='Human36M':
            eval_dataset = Human36M(data_fp, 'test', args, down_transform_list, tmp['face'])
            eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)
            train_dataset = Human36M(data_fp, 'train', args, down_transform_list, tmp['face'])
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=8, drop_last=True)
        else:
            raise Exception('Dataset not support')
        # optimize
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.decay_step, gamma=args.lr_decay)
        # tensorboard
        board = SummaryWriter(osp.join(args.out_dir, 'board'))
        runner.set_train_loader(train_loader, args.epochs, optimizer, scheduler, writer, board, start_epoch=epoch)
        runner.set_eval_loader(eval_loader)
        runner.train()
    elif args.phase == 'eval':
        # dataset
        eval_dataset = FreiHAND(data_fp, 'evaluation', args, tmp['face'])
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)
        runner.set_eval_loader(eval_loader)
        runner.evaluation()
    elif args.phase == 'eval_withgt':
        # dataset
        eval_dataset = Human36M(data_fp, 'test', args, down_transform_list, tmp['face'])
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)
        runner.set_eval_loader(eval_loader)
        runner.evaluation_withgt()
    elif args.phase == 'demo':
        runner.set_demo(args)
        runner.demo()
    else:
        raise Exception('phase error')
