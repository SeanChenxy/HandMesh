import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import random
import numpy as np
from torch.utils.data.dataset import Dataset
from termcolor import cprint
from mobrecon.datasets.comphand import CompHand
from mobrecon.datasets.freihand import FreiHAND
from mobrecon.build import DATA_REGISTRY


@DATA_REGISTRY.register()
class MultipleDatasets(Dataset):
    def __init__(self, cfg, phase='train', writer=None):
        self.cfg = cfg
        self.dbs = []
        if self.cfg.DATA.FREIHAND.USE:
            self.dbs.append( FreiHAND(self.cfg, phase, writer) )
        if self.cfg.DATA.COMPHAND.USE:
            self.dbs.append( CompHand(self.cfg, phase, writer) )
        self.db_num = len(self.dbs)
        self.max_db_data_num = max([len(db) for db in self.dbs])
        self.db_len_cumsum = np.cumsum([len(db) for db in self.dbs])
        self.make_same_len = False
        if writer is not None:
            writer.print_str('Merge train set, total {} samples'.format(self.__len__()))
        cprint('Merge train set, total {} samples'.format(self.__len__()), 'red')

    def __len__(self):
        # all dbs have the same length
        if self.make_same_len:
            return self.max_db_data_num * self.db_num
        # each db has different length
        else:
            return sum([len(db) for db in self.dbs])

    def __getitem__(self, index):
        if self.make_same_len:
            db_idx = index // self.max_db_data_num
            data_idx = index % self.max_db_data_num
            if data_idx >= len(self.dbs[db_idx]) * (self.max_db_data_num // len(self.dbs[db_idx])): # last batch: random sampling
                data_idx = random.randint(0,len(self.dbs[db_idx])-1)
            else: # before last batch: use modular
                data_idx = data_idx % len(self.dbs[db_idx])
        else:
            for i in range(self.db_num):
                if index < self.db_len_cumsum[i]:
                    db_idx = i
                    break
            if db_idx == 0:
                data_idx = index
            else:
                data_idx = index - self.db_len_cumsum[db_idx-1]
        return self.dbs[db_idx][data_idx]

if __name__ == '__main__':
    """Test the dataset
    """
    from mobhand.main import setup
    from options.cfg_options import CFGOptions

    args = CFGOptions().parse()
    args.config_file = 'mobrecon/configs/mobrecon_ds.yml'
    cfg = setup(args)

    dataset = MultipleDatasets(cfg)

    for i in range(0, len(dataset), len(dataset) // 10):
        print(i)
        data = dataset.__getitem__(i)
