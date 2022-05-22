# Copyright (c) Xingyu Chen. All Rights Reserved.

"""
 * @file comphand.py
 * @author chenxingyu (chenxy.sean@gmail.com)
 * @brief CompHand dataset 
 * @version 0.1
 * @date 2022-04-28
 * 
 * @copyright Copyright (c) 2022 chenxingyu
 * 
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import torch.utils.data as data
from utils.fh_utils import *
from utils.vis import base_transform, inv_base_tranmsform, cnt_area
import cv2
from utils.augmentation import Augmentation
from termcolor import cprint
from pathlib import Path
from utils.read import read_mesh
from utils.preprocessing import augmentation, augmentation_2d
from mobrecon.models.loss import contrastive_loss_3d, contrastive_loss_2d
from mobrecon.build import DATA_REGISTRY
import vctoolkit as vc
from mobrecon.tools.kinematics import MPIIHandJoints, mano_to_mpii


@DATA_REGISTRY.register()
class CompHand(data.Dataset):
    def __init__(self, cfg, phase='train', writer=None):
        """Init a CompHand Dataset

        Args:
            cfg : config file
            phase (str, optional): train or eval. Defaults to 'train'.
            writer (optional): log file. Defaults to None.
        """
        super(CompHand, self).__init__()
        self.cfg = cfg
        self.phase = phase
        self.color_aug = Augmentation() if cfg.DATA.COLOR_AUG and 'train' in self.phase else None
        self.j_reg = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../template/j_reg.npy'))
        self.K = np.array([[373.3511425,   0.,        128.],
                           [  0.,        373.3511425, 128.],
                           [  0.,          0.,          1.]])
        self.trans = np.array([ 0.51977897, 6.54544898, 93.32998466])
        path_lib = Path(os.path.join(cfg.DATA.COMPHAND.ROOT))
        self.img_list = sorted(list(path_lib.glob('**/pic256/**/*.png')))
        self.joint_num = 21
        if writer is not None:
            writer.print_str('Loaded CompHand {} {} samples'.format(self.phase, str(len(self.img_list))))
        cprint('Loaded CompHand {} {} samples'.format(self.phase, str(len(self.img_list))), 'red')

    def __getitem__(self, idx):
        if self.cfg.DATA.CONTRASTIVE:
            return self.get_contrastive_sample(idx)
        else:
            return self.get_training_sample(idx)

    def get_contrastive_sample(self, idx):
        """Get contrastive CompHand samples for consistency learning
        """
        # path prepare
        img_path = self.img_list[idx]
        img_name = img_path.parts[-1]
        img_name_split = img_name.split('.')
        num = int(img_name_split[1])
        mesh_path = os.path.join(*img_path.parts[:-2], str(num) + '.obj').replace('pic256', 'model_mano')
        mask_path = os.path.join(mesh_path.replace('obj', 'png').replace('model_mano', 'mask256'))
        img_path = os.path.join(*img_path.parts)

        # read
        img = cv2.imread(img_path)[:, ::-1, ::-1]
        mask = cv2.imread(mask_path)[..., ::-1, 0]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        contours.sort(key=cnt_area, reverse=True)
        bbox = cv2.boundingRect(contours[0])
        center = [bbox[0]+bbox[2]*0.5, bbox[1]+bbox[3]*0.5]
        w, h = bbox[2], bbox[3]
        bbox = [center[0]-0.5 * max(w, h), center[1]-0.5 * max(w, h), max(w, h), max(w, h)]
        vert = read_mesh(mesh_path).x.numpy()
        vert[:, 0] *= -1   # flip
        joint_cam = mano_to_mpii(np.dot(self.j_reg, vert))
        K = self.K.copy()
        joint_img = projectPoints(joint_cam, K)
        princpt = K[0:2, 2].astype(np.float32)
        focal = np.array( [K[0, 0], K[1, 1]], dtype=np.float32)

        # multiple aug
        roi_list = []
        calib_list = []
        mask_list = []
        vert_list = []
        joint_cam_list = []
        joint_img_list = []
        aug_param_list = []
        bb2img_trans_list = []
        for _ in range(2):
            # augmentation
            roi, img2bb_trans, bb2img_trans, aug_param, do_flip, scale, roi_mask = augmentation(img, bbox, self.phase, exclude_flip=not self.cfg.DATA.COMPHAND.FLIP, input_img_shape=(self.cfg.DATA.SIZE, self.cfg.DATA.SIZE), mask=mask,
                                                                                            base_scale=self.cfg.DATA.COMPHAND.BASE_SCALE, scale_factor=self.cfg.DATA.COMPHAND.SCALE, rot_factor=self.cfg.DATA.COMPHAND.ROT,
                                                                                            shift_wh=[bbox[2], bbox[3]], gaussian_std=self.cfg.DATA.STD)
            if self.color_aug is not None:
                roi = self.color_aug(roi)
            roi = base_transform(roi, self.cfg.DATA.SIZE, mean=self.cfg.DATA.IMG_MEAN, std=self.cfg.DATA.IMG_STD)
            roi = torch.from_numpy(roi).float()
            roi_mask = torch.from_numpy(roi_mask).float()
            bb2img_trans = torch.from_numpy(bb2img_trans).float()
            aug_param = torch.from_numpy(aug_param).float()

            # joints
            joint_img_, princpt_ = augmentation_2d(img, joint_img, princpt, img2bb_trans, do_flip)
            joint_img_ = torch.from_numpy(joint_img_[:, :2]).float() / self.cfg.DATA.SIZE

            # 3D rot
            rot = aug_param[0].item()
            rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                                    [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                                    [0, 0, 1]], dtype=np.float32)
            joint_cam_ = torch.from_numpy(np.dot(rot_aug_mat, joint_cam.T).T).float()
            vert_ = torch.from_numpy(np.dot(rot_aug_mat, vert.T).T).float()

            # K
            focal_ = focal * roi.size(1) / (bbox[2]*aug_param[1])
            calib = np.eye(4)
            calib[0, 0] = focal_[0]
            calib[1, 1] = focal_[1]
            calib[:2, 2:3] = princpt_[:, None]
            calib = torch.from_numpy(calib).float()

            roi_list.append(roi)
            mask_list.append(roi_mask.unsqueeze(0))
            calib_list.append(calib)
            vert_list.append(vert_)
            joint_cam_list.append(joint_cam_)
            joint_img_list.append(joint_img_)
            aug_param_list.append(aug_param)
            bb2img_trans_list.append(bb2img_trans)

        roi = torch.cat(roi_list, 0)
        mask = torch.cat(mask_list, 0)
        calib = torch.cat(calib_list, 0)
        joint_cam = torch.cat(joint_cam_list, -1)
        vert = torch.cat(vert_list, -1)
        joint_img = torch.cat(joint_img_list, -1)
        aug_param = torch.cat(aug_param_list, 0)
        bb2img_trans = torch.cat(bb2img_trans_list, -1)

        # postprocess root and joint_cam
        root = joint_cam[0].clone()
        joint_cam -= root
        vert -= root
        joint_cam /= 0.2
        vert /= 0.2

        # out
        res = {'img': roi, 'joint_img': joint_img, 'joint_cam': joint_cam, 'verts': vert, 'mask': mask,
               'root': root, 'calib': calib, 'aug_param': aug_param, 'bb2img_trans': bb2img_trans,}
        return res

    def get_training_sample(self, idx):
        """Get a CompHand sample for training
        """
        # path prepare
        img_path = self.img_list[idx]
        img_name = img_path.parts[-1]
        img_name_split = img_name.split('.')
        num = int(img_name_split[1])
        mesh_path = os.path.join(*img_path.parts[:-2], str(num) + '.obj').replace('pic256', 'model_mano')
        mask_path = os.path.join(mesh_path.replace('obj', 'png').replace('model_mano', 'mask256'))
        img_path = os.path.join(*img_path.parts)

        # read
        img = cv2.imread(img_path)[:, ::-1, ::-1]
        mask = cv2.imread(mask_path)[..., ::-1, 0]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        contours.sort(key=cnt_area, reverse=True)
        bbox = cv2.boundingRect(contours[0])
        center = [bbox[0]+bbox[2]*0.5, bbox[1]+bbox[3]*0.5]
        w, h = bbox[2], bbox[3]
        bbox = [center[0]-0.5 * max(w, h), center[1]-0.5 * max(w, h), max(w, h), max(w, h)]
        vert = read_mesh(mesh_path).x.numpy()
        vert[:, 0] *= -1   # flip
        joint_cam = mano_to_mpii(np.dot(self.j_reg, vert))
        K = self.K.copy()
        joint_img = projectPoints(joint_cam, K)
        princpt = K[0:2, 2].astype(np.float32)
        focal = np.array( [K[0, 0], K[1, 1]], dtype=np.float32)

        # augmentation
        roi, img2bb_trans, bb2img_trans, aug_param, do_flip, scale, mask = augmentation(img, bbox, self.phase, exclude_flip=not self.cfg.DATA.COMPHAND.FLIP, input_img_shape=(self.cfg.DATA.SIZE, self.cfg.DATA.SIZE), mask=mask,
                                                                                     base_scale=self.cfg.DATA.COMPHAND.BASE_SCALE, scale_factor=self.cfg.DATA.COMPHAND.SCALE, rot_factor=self.cfg.DATA.COMPHAND.ROT,
                                                                                     shift_wh=[bbox[2], bbox[3]], gaussian_std=self.cfg.DATA.STD)
        if self.color_aug is not None:
            roi = self.color_aug(roi)
        roi = base_transform(roi, self.cfg.DATA.SIZE, mean=self.cfg.DATA.IMG_MEAN, std=self.cfg.DATA.IMG_STD)
        # img = inv_based_tranmsform(roi)
        # cv2.imshow('test', img)
        # cv2.waitKey(0)
        roi = torch.from_numpy(roi).float()
        mask = torch.from_numpy(mask).float()
        bb2img_trans = torch.from_numpy(bb2img_trans).float()

        # joints
        joint_img, princpt = augmentation_2d(img, joint_img, princpt, img2bb_trans, do_flip)
        joint_img = torch.from_numpy(joint_img[:, :2]).float() / self.cfg.DATA.SIZE

        # 3D rot
        rot = aug_param[0]
        rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                                [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                                [0, 0, 1]], dtype=np.float32)
        joint_cam = np.dot(rot_aug_mat, joint_cam.T).T
        vert = np.dot(rot_aug_mat, vert.T).T

        # K
        focal = focal * roi.size(1) / (bbox[2]*aug_param[1])
        calib = np.eye(4)
        calib[0, 0] = focal[0]
        calib[1, 1] = focal[1]
        calib[:2, 2:3] = princpt[:, None]
        calib = torch.from_numpy(calib).float()

        # postprocess root and joint_cam
        root = joint_cam[0].copy()
        joint_cam -= root
        vert -= root
        joint_cam /= 0.2
        vert /= 0.2
        root = torch.from_numpy(root).float()
        joint_cam = torch.from_numpy(joint_cam).float()
        vert = torch.from_numpy(vert).float()

        res = {'img': roi, 'joint_img': joint_img, 'joint_cam': joint_cam, 'verts': vert, 'mask': mask, 'root': root, 'calib': calib}
        return res

    def __len__(self):
        return len(self.img_list)

    def visualization(self, res, idx):
        """Visualization of correctness
        """
        import matplotlib.pyplot as plt
        from mobrecon.tools.vis import perspective
        num_sample = (1, 2)[self.cfg.DATA.CONTRASTIVE]
        for i in range(num_sample):
            fig = plt.figure(figsize=(8, 2))
            img = inv_base_tranmsform(res['img'].numpy()[i*3:(i+1)*3])
            # joint_img
            if 'joint_img' in res:
                ax = plt.subplot(1, 4, 1)
                vis_joint_img = vc.render_bones_from_uv(np.flip(res['joint_img'].numpy()[:, i*2:(i+1)*2]*self.cfg.DATA.SIZE, axis=-1).copy(),
                                                        img.copy(), MPIIHandJoints, thickness=2)
                ax.imshow(vis_joint_img)
                ax.set_title('kps2d')
                ax.axis('off')
            # aligned joint_cam
            if 'joint_cam' in res:
                ax = plt.subplot(1, 4, 2)
                xyz = res['joint_cam'].numpy()[:, i*3:(i+1)*3].copy()
                root = res['root'].numpy()[i*3:(i+1)*3].copy()
                xyz = xyz * 0.2 + root
                proj3d = perspective(torch.from_numpy(xyz.copy()).permute(1, 0).unsqueeze(0), res['calib'][i*4:(i+1)*4].unsqueeze(0))[0].numpy().T
                vis_joint_img = vc.render_bones_from_uv(np.flip(proj3d[:, :2], axis=-1).copy(),
                                                        img.copy(), MPIIHandJoints, thickness=2)
                ax.imshow(vis_joint_img)
                ax.set_title('kps3d2d')
                ax.axis('off')
            # aligned verts
            if 'verts' in res:
                ax = plt.subplot(1, 4, 3)
                vert = res['verts'].numpy()[:, i*3:(i+1)*3].copy()
                vert = vert * 0.2 + root
                proj_vert = perspective(torch.from_numpy(vert.copy()).permute(1, 0).unsqueeze(0), res['calib'][i*4:(i+1)*4].unsqueeze(0))[0].numpy().T
                ax.imshow(img)
                plt.plot(proj_vert[:, 0], proj_vert[:, 1], 'o', color='red', markersize=1)
                ax.set_title('verts')
                ax.axis('off')
            # mask
            if 'mask' in res:
                ax = plt.subplot(1, 4, 4)
                if res['mask'].ndim == 3:
                    mask = res['mask'].numpy()[i] * 255
                else:
                    mask = res['mask'].numpy() * 255
                mask_ = np.concatenate([mask[:, :, None]] + [np.zeros_like(mask[:, :, None])] * 2, 2).astype(np.uint8)
                img_mask = cv2.addWeighted(img, 1, mask_, 0.5, 1)
                ax.imshow(img_mask)
                ax.set_title('mask')
                ax.axis('off')
            if self.cfg.DATA.CONTRASTIVE:
                aug_param = data['aug_param'].unsqueeze(0)
                vert = data['verts'].unsqueeze(0)
                joint_img = data['joint_img'].unsqueeze(0)
                uv_trans = data['bb2img_trans'].unsqueeze(0)
                loss3d = contrastive_loss_3d(vert, aug_param)
                loss2d = contrastive_loss_2d(joint_img, uv_trans, data['img'].size(2))
                print(idx, loss3d, loss2d)
            plt.subplots_adjust(left=0., right=0.95, top=0.95, bottom=0.03, wspace=0.12, hspace=0.1)
            plt.show()

if __name__ == '__main__':
    """Test the dataset
    """
    from mobrecon.main import setup
    from options.cfg_options import CFGOptions

    args = CFGOptions().parse()
    args.config_file = 'mobrecon/configs/mobrecon_ds.yml'
    cfg = setup(args)

    dataset = CompHand(cfg, 'train')
    for i in range(0, len(dataset), len(dataset)//10):
        print(i)
        data = dataset.__getitem__(i)
        dataset.visualization(data, i)
