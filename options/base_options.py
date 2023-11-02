import argparse

class BaseOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--local_rank', type=int, default=0)
        parser.add_argument('--exp_name', type=str, default='test')
        parser.add_argument('--n_threads', type=int, default=4)
        parser.add_argument('--device_idx', type=int, nargs='+', default=[-1,])

        # dataset hyperparameters
        parser.add_argument('--dataset', type=str, default='FreiHAND')
        parser.add_argument('--pos_aug', type=float, default=3)
        parser.add_argument('--rot_aug', type=float, default=90)
        parser.add_argument('--color_aug', type=self.str2bool, default='yes')
        parser.add_argument('--size', type=int, default=224)
        parser.add_argument('--ms_mesh', type=self.str2bool, default='yes')

        # network hyperparameters
        parser.add_argument('--out_channels', nargs='+', default=[64, 128, 256, 512], type=int)
        parser.add_argument('--ds_factors', nargs='+', default=[2, 2, 2, 2], type=float)
        parser.add_argument('--in_channels', type=int, default=3)
        parser.add_argument('--seq_length', type=int, default=[27, 27, 27, 27], nargs='+')
        parser.add_argument('--dilation', type=int, default=[1, 1, 1, 1], nargs='+')
        parser.add_argument('--model', type=str, default='cmr_sg')
        parser.add_argument('--backbone', type=str, default='ResNet18')
        parser.add_argument('--bn', type=self.str2bool, default='no')
        parser.add_argument('--att', type=self.str2bool, default='no')
        parser.add_argument('--dsconv', type=self.str2bool, default='no')

        # optimizer hyperparmeters
        parser.add_argument('--optimizer', type=str, default='Adam')
        parser.add_argument('--lr_scheduled', type=str, default='MultiStep')
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--lr_decay', type=float, default=0.1)
        parser.add_argument('--decay_step', type=int, nargs='+', default=[30, ])
        parser.add_argument('--weight_decay', type=float, default=0)

        # training hyperparameters
        parser.add_argument('--phase', type=str, default='train')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--epochs', type=int, default=38)
        parser.add_argument('--resume', type=str, default='')

        # others
        # parser.add_argument('--seed', type=int, default=1)

        self.initialized = True
        return parser

    def str2bool(self, v):
        return v.lower() in ("yes", "true", "t", "1")

    def parse(self):

        parser = argparse.ArgumentParser(description='mesh generator')
        self.initialize(parser)
        args = parser.parse_args()

        return args
