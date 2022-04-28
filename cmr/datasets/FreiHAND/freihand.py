import torch
import torch.utils.data as data
from utils.fh_utils import load_db_annotation, read_img, read_img_abs, read_msk, projectPoints, read_mesh
from utils.vis import base_transform, inv_base_tranmsform, cnt_area, uv2map
from termcolor import cprint
from utils.vis import crop_roi
from utils.augmentation import Augmentation, crop_roi, rotate, get_m1to1_gaussian_rand
from cmr.models.network import Pool
import pickle
import cv2
import os
import numpy as np


class FreiHAND(data.Dataset):
    def __init__(self, root, phase, args, faces, writer=None, down_sample_list=None, img_std=0.5, img_mean=0.5, ms=True):
        super(FreiHAND, self).__init__()
        self.root = root
        self.phase = phase
        self.down_sample_list = down_sample_list
        self.size = args.size
        self.faces = faces
        self.img_std = img_std
        self.img_mean = img_mean
        self.ms = ms
        self.pos_aug = args.pos_aug if 'train' in self.phase else 0
        self.rot_aug = args.rot_aug if 'train' in self.phase else 0
        assert 0 <= self.rot_aug <= 180, 'rotaion limit must be in [0, 180]'
        self.color_aug = Augmentation(size=self.size) if args.color_aug and 'train' in self.phase else None
        with open(os.path.join(args.work_dir, '../template/MANO_RIGHT.pkl'), 'rb') as f:
            mano = pickle.load(f, encoding='latin1')
        self.j_regressor = np.zeros([21, 778])
        self.j_regressor[:16] = mano['J_regressor'].toarray()
        for k, v in {16: 333, 17: 444, 18: 672, 19: 555, 20: 744}.items():
            self.j_regressor[k, v] = 1
        self.std = torch.tensor(0.20)
        self.db_data_anno = tuple(load_db_annotation(self.root, writer, self.phase))
        self.one_version_len = len(self.db_data_anno)
        if 'train' in self.phase:
            self.db_data_anno *= 4
        cprint('Loaded FreiHand {} {} samples'.format(self.phase, str(len(self.db_data_anno))), 'red')

    def __getitem__(self, idx):
        if self.phase == 'training':
            return self.get_training_sample(idx)
        elif self.phase == 'evaluation':
            return self.get_eval_sample(idx)
        else:
            raise Exception('phase error')

    def get_training_sample(self, idx):
        img = read_img_abs(idx, self.root, 'training')
        mask = read_msk(idx % self.one_version_len, self.root, 'training')
        K, mano, xyz = self.db_data_anno[idx]
        K, xyz, mano = np.array(K), np.array(xyz), np.array(mano)
        uv = projectPoints(xyz, K)
        v0 = read_mesh(idx % self.one_version_len, self.root).x.numpy()
        img, K, uv, mask, v0, xyz = self.crop_data(img, K, uv=uv, mask=mask, v0=v0, xyz=xyz)
        if self.color_aug is not None:
            img = self.color_aug(img)
        img = base_transform(img, size=self.size, mean=self.img_mean, std=self.img_std)
        uv_map = uv2map(uv.astype(np.int), img.shape[1:])
        uv_map = cv2.resize(uv_map.transpose(1, 2, 0), (img.shape[2]//2, img.shape[1]//2)).transpose(2, 0, 1)
        mask = cv2.resize(mask, (img.shape[2]//2, img.shape[1]//2))
        img, mask, K, xyz, uv, uv_map, v0, mano = [torch.from_numpy(x).float() for x in [img, mask, K, xyz, uv, uv_map, v0, mano[0, :58]]]

        xyz_root = xyz[0]
        v0 = (v0 - xyz_root) / self.std
        xyz = (xyz - xyz_root) / self.std
        if self.ms:
            v1 = Pool(v0.unsqueeze(0), self.down_sample_list[0])[0]
            v2 = Pool(v1.unsqueeze(0), self.down_sample_list[1])[0]
            v3 = Pool(v2.unsqueeze(0), self.down_sample_list[2])[0]
            gt = [v0, v1, v2, v3]
        else:
            gt = [v0, ]

        data = {'img': img,
                'mesh_gt': gt,
                'K': K,
                'mask_gt': mask,
                'xyz_gt': xyz,
                'uv_point': uv,
                'uv_gt': uv_map,
                'xyz_root': xyz_root,
                }

        return data

    def get_eval_sample(self, idx):
        img = read_img(idx, self.root, 'evaluation', 'gs')
        K, _ = self.db_data_anno[idx]
        K = np.array(K)
        img, K = self.crop_data(img, K)
        img = base_transform(img, self.size, mean=self.img_mean, std=self.img_std)
        img = torch.from_numpy(img)
        K = torch.from_numpy(K)

        return {'img': img,
                'K': K,
                }

    def __len__(self):

        return len(self.db_data_anno)

    def get_face(self):
        return self.faces

    def crop_data(self, img, K, uv=None, mask=None, v0=None, xyz=None):
        if 'train' in self.phase:
            assert mask is not None and v0 is not None and xyz is not None
            if self.rot_aug > 0:
                angle = np.random.randint(-self.rot_aug, self.rot_aug)
                rot_mapping = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), angle, 1.0)  # 12
                img = rotate(img, rot_mapping)
                mask = rotate(mask, rot_mapping)
                rot_point = np.array([[np.cos(angle / 180. * np.pi), np.sin(angle / 180. * np.pi), 0],
                                      [-np.sin(angle / 180. * np.pi), np.cos(angle / 180. * np.pi), 0],
                                      [0, 0, 1]])
                uv = np.matmul(rot_point[:2, :2], (uv - np.array([[img.shape[1] // 2, img.shape[0] // 2]])).T).T + np.array(
                    [[img.shape[1] // 2, img.shape[0] // 2]])
                v0 = np.matmul(rot_point, v0.T).T
                xyz = np.matmul(rot_point, xyz.T).T
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = list(contours)
            contours.sort(key=cnt_area, reverse=True)
            x, y, w, h = cv2.boundingRect(contours[0])
            center = (x + w / 2, y + h / 2)
            if self.pos_aug > 0:
                amplify = get_m1to1_gaussian_rand(self.pos_aug)
                shift_x = get_m1to1_gaussian_rand(self.pos_aug)
                shift_y = get_m1to1_gaussian_rand(self.pos_aug)
            else:
                amplify = shift_x = shift_y = 0
            size = max(w, h) / 2 * (1.3 + amplify * 0.2)
            shift_lim = (size - w / 2, size - h / 2)
            x_shift = shift_x * shift_lim[0]
            y_shift = shift_y * shift_lim[1]
            tl_x = int(center[0] - size + x_shift)
            tl_y = int(center[1] - size + y_shift)
            br_x = int(center[0] + size + x_shift)
            br_y = int(center[1] + size + y_shift)
        else:
            size = 100 / 2 * 1.3
            tl_x = int(img.shape[1]/2-size)
            tl_y = int(img.shape[0]/2-size)
            br_x = int(img.shape[1]/2+size)
            br_y = int(img.shape[0]/2+size)

        scale = self.size / 2 / size
        img = crop_roi(img, (tl_x, tl_y, br_x, br_y), self.size)
        if mask is not None:
            mask = crop_roi(mask, (tl_x, tl_y, br_x, br_y), self.size)
        if uv is not None:
            uv[:, 0] = (uv[:, 0] - tl_x) * scale
            uv[:, 1] = (uv[:, 1] - tl_y) * scale
        K[0, 0] *= scale
        K[1, 1] *= scale
        K[0, 2] = scale*(K[0, 2]-tl_x+0.5) - 0.5
        K[1, 2] = scale*(K[1, 2]-tl_y+0.5) - 0.5
        ret = [img, K]
        if uv is not None:
            ret.append(uv)
        if mask is not None:
            ret.append(mask)
        if v0 is not None:
            ret.append(v0)
        if xyz is not None:
            ret.append(xyz)
        return ret



if __name__ == '__main__':
    import pickle
    import utils
    from utils import utils
    from options.base_options import BaseOptions
    import cv2

    args = BaseOptions().parse()
    with open('../../template/transform.pkl', 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')
    down_transform_list = [
        utils.to_sparse(down_transform)
        for down_transform in tmp['down_transform']
    ]

    args.phase = 'training'
    args.size = 224
    args.work_dir = os.path.join( os.path.dirname(os.path.realpath(__file__)), '../..' )
    dataset = FreiHAND('../../data/FreiHAND', args.phase, args, tmp['face'], writer=None,
                       down_sample_list=down_transform_list, ms=args.ms_mesh)
    for i in range(len(dataset)):
        data = dataset.get_training_sample(i)
