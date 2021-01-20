import argparse

class BaseOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--local_rank', type=int, default=0)
        parser.add_argument('--exp_name', type=str, default='cmr_sp')
        parser.add_argument('--dataset', type=str, default='FreiHAND')
        parser.add_argument('--size', type=int, default=224)
        parser.add_argument('--n_threads', type=int, default=4)
        parser.add_argument('--device_idx', type=int, nargs='+', default=[-1,])

        # network hyperparameters
        parser.add_argument('--out_channels', nargs='+', default=[64, 128, 256, 512], type=int)
        parser.add_argument('--ds_factors', nargs='+', default=[2, 2, 2, 2], type=float)
        parser.add_argument('--in_channels', type=int, default=3)
        parser.add_argument('--seq_length', type=int, default=[27, 27, 27, 27], nargs='+')
        parser.add_argument('--dilation', type=int, default=[1, 1, 1, 1], nargs='+')
        parser.add_argument('--model', type=str, default='cmr_sp')
        parser.add_argument('--backbone', type=str, default='ResNet50')

        # training hyperparameters
        parser.add_argument('--phase', type=str, default='demo')
        parser.add_argument('--resume', type=str, default='cmr_sp_res50_freihand.pt')

        self.initialized = True
        return parser

    def str2bool(self, v):
        return v.lower() in ("yes", "true", "t", "1")

    def parse(self):

        parser = argparse.ArgumentParser(description='mesh generator')
        self.initialize(parser)
        args = parser.parse_args()

        return args
