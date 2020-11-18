import torch
import torch.utils.data as data
from utils.fh_utils import *
from utils.vis import base_transform, inv_base_transform
from termcolor import cprint
from utils.vis import crop_roi


class FreiHAND(data.Dataset):

    def __init__(self, root, phase, args, faces):
        super(FreiHAND, self).__init__()
        self.root = root
        self.phase = phase
        self.size = args.size
        self.faces = faces
        self.std = torch.tensor(0.20)
        self.db_data_anno = tuple(load_db_annotation(self.root, None, self.phase))
        cprint('Loaded FreiHand {} {} samples'.format(self.phase, str(len(self.db_data_anno))), 'red')

    def __getitem__(self, idx):
        if self.phase == 'evaluation':
            return self.get_eval_sample(idx)
        else:
            raise Exception('phase error')

    def get_eval_sample(self, idx):
        img = read_img(idx, self.root, 'evaluation', 'gs')
        K, _ = self.db_data_anno[idx]
        K = np.array(K)
        img, K = self.crop_data(img, K)
        img = base_transform(img, self.size)
        img = torch.from_numpy(img)
        K = torch.from_numpy(K)

        return {'img': img,
                'K': K,
                }

    def __len__(self):

        return len(self.db_data_anno)

    def get_face(self):
        return self.faces

    def crop_data(self, img, K):
        size = 100 / 2 * 1.3
        tl_x = int(img.shape[1]/2-size)
        tl_y = int(img.shape[0]/2-size)
        br_x = int(img.shape[1]/2+size)
        br_y = int(img.shape[0]/2+size)

        scale = self.size / 2 / size
        img = crop_roi(img, (tl_x, tl_y, br_x, br_y), self.size)
        K[0, 0] *= scale
        K[1, 1] *= scale
        K[0, 2] = scale*(K[0, 2]-tl_x+0.5) - 0.5
        K[1, 2] = scale*(K[1, 2]-tl_y+0.5) - 0.5

        return img, K


if __name__ == '__main__':
    import pickle
    from options.base_options import BaseOptions
    import cv2

    args = BaseOptions().parse()
    with open('../../template/transform.pkl', 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')

    args.phase = 'evaluation'
    args.size = 224
    dataset = FreiHAND('../../data/FreiHAND', args.phase, args, tmp['face'])
    for i in range(len(dataset)):
        data = dataset.get_eval_sample(i)
        cv2.imshow('test', inv_base_transform(data['img'].numpy())[:, :, ::-1])
        cv2.waitKey(0)

