# Copyright (c) Liuhao Ge. All Rights Reserved.
r"""
Real world test set
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import scipy.io as sio
import os.path as osp
import cv2
import numpy as np
import torch
import torch.utils.data
from utils.vis import base_transform, inv_base_tranmsform, uv2map
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from termcolor import cprint
from mobrecon.build import DATA_REGISTRY
from mobrecon.tools.vis import perspective

@DATA_REGISTRY.register()
class Ge(torch.utils.data.Dataset):
    def __init__(self, cfg, phase='eval', writer=None):
        self.cfg = cfg
        self.phase = phase
        self.mean = torch.tensor([0.0016, 0.0025, 0.7360]).float()
        self.std = torch.tensor(0.20)
        self.img_std = self.cfg.DATA.IMG_STD
        self.img_mean = self.cfg.DATA.IMG_MEAN
        self.root = self.cfg.DATA.GE.ROOT
        self.size = self.cfg.DATA.SIZE

        mat_params = sio.loadmat(os.path.join(self.root, 'params.mat'))
        self.image_paths = mat_params["image_path"]

        self.cam_params = torch.from_numpy(mat_params["cam_param"]).float()  # N x 4, [fx, fy, u0, v0]
        assert len(self.image_paths) == self.cam_params.shape[0]

        self.bboxes = torch.from_numpy(mat_params["bbox"]).float()  # N x 4, bounding box in the original image, [x, y, w, h]
        assert len(self.image_paths) == self.bboxes.shape[0]

        self.pose_roots = torch.from_numpy(mat_params["pose_root"]).float()  # N x 3, [root_x, root_y, root_z]
        assert len(self.image_paths) == self.pose_roots.shape[0]

        if "pose_scale" in mat_params.keys():
            self.pose_scales = torch.from_numpy(mat_params["pose_scale"]).squeeze().float()  # N, length of first bone of middle finger
        else:
            self.pose_scales = torch.ones(len(self.image_paths)) * 5.0
        assert len(self.image_paths) == self.pose_scales.shape[0]

        mat_gt = sio.loadmat(os.path.join(self.root, 'pose_gt.mat'))
        self.pose_gts = torch.from_numpy(mat_gt["pose_gt"])  # N x K x 3
        assert len(self.image_paths) == self.pose_gts.shape[0]

        if writer is not None:
            writer.print_str('Loaded Ge test {} samples'.format(len(self.image_paths)))
        cprint('Loaded Ge test {} samples'.format(len(self.image_paths)), 'red')


    def __getitem__(self, idx):
        img = cv2.imread(osp.join(self.root, self.image_paths[idx]))[:, ::-1, ::-1]
        img = base_transform(img, self.size, std=self.img_std, mean=self.img_mean)
        bbox = self.bboxes[idx].clone()
        bbox[0] = 1280 - bbox[0] - bbox[2]
        xyz = self.pose_gts[idx].clone() / 100
        xyz[:, 0] *= -1

        xyz_root = self.pose_roots[idx].clone().unsqueeze(0) / 100
        xyz_root[:, 0] *= -1

        fx, fy, u0, v0 = self.cam_params[idx].clone()
        u0 = 1280 - u0
        scale = self.size / bbox[2]
        calib = np.eye(4)
        calib[0, 0] = fx * scale
        calib[1, 1] = fy * scale
        calib[0, 2] = scale * (u0 - bbox[0] + 0.5) - 0.5
        calib[1, 2] = scale * (v0 - bbox[1] + 0.5) - 0.5
        calib = torch.from_numpy(calib).float()
        uv = perspective(xyz.clone().T.unsqueeze(0), calib.unsqueeze(0))[0].numpy().T[:, :2]
        uv_map = uv2map(uv.astype(np.int32), img.shape[1:])
        uv_map = cv2.resize(uv_map.transpose(1, 2, 0), (img.shape[2]//2, img.shape[1]//2)).transpose(2, 0, 1)
        uv = uv / img.shape[1:][::-1]
        xyz -= xyz_root
        img, uv_point, uv_map = [torch.from_numpy(x).float() for x in [img, uv, uv_map]]

        res = {'img': img, 'joint_img': uv_point, 'joint_cam': xyz, 'root': xyz_root, 'calib': calib, 'joint_img_map': uv_map}

        return res

    def visualization(self, idx, data):
        gs = gridspec.GridSpec(1, 2)
        # xyz = (data['xyz_gt'] * self.std + data['xyz_root']).numpy()
        fig = plt.figure()
        ax = fig.add_subplot(gs[0, 0])
        img = inv_base_tranmsform(data['img'].numpy(), std=self.img_std, mean=self.img_mean)
        ax.imshow(img)
        uv_point = data['joint_img'].numpy() * img.shape[:2][::-1]
        ax.scatter(uv_point[:, 0], uv_point[:, 1])
        ax.axis('off')
        ax = fig.add_subplot(gs[0, 1])
        uv_map = cv2.resize((data['joint_img_map'].sum(dim=0).clamp(max=1).numpy()*255).astype(np.uint8), (self.size, self.size))
        uv_map_ = np.concatenate([uv_map[:, :, None]] + [np.zeros_like(uv_map[:, :, None])] * 2, 2)
        img_uv = cv2.addWeighted(img, 1, uv_map_, 0.5, 1)
        ax.imshow(img_uv)
        ax.axis('off')
        # plt.savefig(str(idx) + '.jpg')
        plt.show()

    def __len__(self):
        return len(self.image_paths)

    def evaluate_pose(self, results_pose_cam_xyz, save_results=False, output_dir=""):
        avg_est_error = 0.0
        for image_id, est_pose_cam_xyz in results_pose_cam_xyz.items():
            dist = est_pose_cam_xyz - self.pose_gts[image_id]  # K x 3
            avg_est_error += dist.pow(2).sum(-1).sqrt().mean()

        avg_est_error /= len(results_pose_cam_xyz)

        if save_results:
            eval_results = {}
            image_ids = results_pose_cam_xyz.keys()
            image_ids.sort()
            eval_results["image_ids"] = np.array(image_ids)
            eval_results["gt_pose_xyz"] = [self.pose_gts[image_id].unsqueeze(0) for image_id in image_ids]
            eval_results["est_pose_xyz"] = [results_pose_cam_xyz[image_id].unsqueeze(0) for image_id in image_ids]
            eval_results["gt_pose_xyz"] = torch.cat(eval_results["gt_pose_xyz"], 0).numpy()
            eval_results["est_pose_xyz"] = torch.cat(eval_results["est_pose_xyz"], 0).numpy()
            sio.savemat(osp.join(output_dir, "pose_estimations.mat"), eval_results)

        return avg_est_error.item()

if __name__ == '__main__':
    """Test the dataset
    """
    from mobrecon.main import setup
    from options.cfg_options import CFGOptions

    args = CFGOptions().parse()
    args.config_file = 'mobrecon/configs/mobrecon_ds.yml'
    cfg = setup(args)

    dataset = Ge(cfg, 'eval')
    l = []
    for i in range(0, 500, 50):
        print(i)
        data = dataset.__getitem__(i)
        dataset.visualization(i, data)

