import sys
sys.path.insert(0, '/home/chenxingyu/Documents/cmr_demo_pytorch')
import os
import os.path as osp
import numpy as np
import torch
import cv2
import random
import json
import math
import copy
import transforms3d
from pycocotools.coco import COCO
from utils.smpl import SMPL
from utils.preprocessing import load_img, get_bbox, process_bbox, generate_patch_image, augmentation
from utils.transforms import world2cam, cam2pixel, pixel2cam, rigid_align, transform_joint_to_other_db
from utils.vis import base_transform, inv_base_tranmsform, uv2map
from utils.read import save_obj, save_mesh
from utils.read import read_mesh as read_mesh_
from utils.fh_utils import projectPoints, plot_hand
from cmr.models.network import Pool
from utils.augmentation import Augmentation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.utils.data as data
from termcolor import cprint


class Human36M(data.Dataset):
    def __init__(self, root, data_split, args, down_sample_list, faces):
        self.root = root
        self.data_split = data_split
        self.size = args.size
        self.faces = faces
        self.down_sample_list = down_sample_list
        self.std = torch.tensor(1.0)
        self.color_aug = Augmentation(size=self.size) if args.color_aug and 'train' in self.data_split else None
        self.img_dir = osp.join(self.root, 'images')
        self.annot_path = osp.join(self.root, 'annotations')
        self.human_bbox_root_dir = osp.join(self.root, 'Human36M', 'rootnet_output', 'bbox_root_human36m_output.json')
        self.action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases',
                            'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog',
                            'WalkTogether']
        self.fitting_thr = 25  # milimeter

        # H36M joint set
        self.h36m_joint_num = 17
        self.h36m_joints_name = (
        'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head_top',
        'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.h36m_flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        self.h36m_skeleton = (
        (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2),
        (2, 3), (0, 4), (4, 5), (5, 6))
        self.h36m_root_joint_idx = self.h36m_joints_name.index('Pelvis')
        self.h36m_eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
        self.h36m_joint_regressor = np.load(osp.join(self.root, 'J_regressor_h36m_correct.npy'))

        # SMPL joint set
        self.smpl = SMPL(self.root)
        self.face = self.smpl.face # self.faces[0]
        self.joint_regressor = self.smpl.joint_regressor
        self.vertex_num = self.smpl.vertex_num # 6890
        self.joint_num = self.smpl.joint_num # 29
        self.joints_name = self.smpl.joints_name
        self.flip_pairs = self.smpl.flip_pairs
        self.skeleton = self.smpl.skeleton
        self.root_joint_idx = self.smpl.root_joint_idx
        self.face_kps_vertex = self.smpl.face_kps_vertex

        self.datalist = self.load_data()

    def get_subsampling_ratio(self):
        if self.data_split == 'train':
            return 5
        elif self.data_split == 'test':
            return 64
        else:
            assert 0, print('Unknown subset')

    def get_subject(self):
        if self.data_split == 'train':
            subject = [1, 5, 6, 7, 8]
        elif self.data_split == 'test':
            subject = [9, 11]
        else:
            assert 0, print("Unknown subset")

        return subject

    def load_data(self):
        subject_list = self.get_subject()
        sampling_ratio = self.get_subsampling_ratio()

        # aggregate annotations from each subject
        db = COCO()
        cameras = {}
        joints = {}
        smpl_params = {}
        for subject in subject_list:
            # data load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_data.json'), 'r') as f:
                annot = json.load(f)
            if len(db.dataset) == 0:
                for k, v in annot.items():
                    db.dataset[k] = v
            else:
                for k, v in annot.items():
                    db.dataset[k] += v
            # camera load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_camera.json'), 'r') as f:
                cameras[str(subject)] = json.load(f)
            # joint coordinate load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_joint_3d.json'), 'r') as f:
                joints[str(subject)] = json.load(f)
            # smpl parameter load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_smpl_param.json'), 'r') as f:
                smpl_params[str(subject)] = json.load(f)
        db.createIndex()

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_dir, img['file_name'])
            img_shape = (img['height'], img['width'])

            # check subject and frame_idx
            frame_idx = img['frame_idx']
            if frame_idx % sampling_ratio != 0:
                continue

            # check smpl parameter exist
            subject = img['subject']
            action_idx = img['action_idx']
            subaction_idx = img['subaction_idx']
            frame_idx = img['frame_idx']
            try:
                smpl_param = smpl_params[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)]
            except KeyError:
                smpl_param = None
                continue

            # camera parameter
            cam_idx = img['cam_idx']
            cam_param = cameras[str(subject)][str(cam_idx)]
            R, t, f, c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(
                cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)
            cam_param = {'R': R, 't': t, 'focal': f, 'princpt': c}

            # only use frontal camera following previous works (HMR and SPIN)
            if self.data_split == 'test' and str(cam_idx) != '4':
                continue

            # project world coordinate to cam, image coordinate space
            joint_world = np.array(joints[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)], dtype=np.float32)
            joint_cam = world2cam(joint_world, R, t)
            joint_img = cam2pixel(joint_cam, f, c)
            joint_valid = np.ones((self.h36m_joint_num, 1))

            bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'])
            if bbox is None: continue
            root_joint_depth = joint_cam[self.h36m_root_joint_idx][2]

            datalist.append({
                'img_path': img_path,
                'img_id': image_id,
                'img_shape': img_shape,
                'bbox': bbox,
                'joint_img': joint_img,
                'joint_cam': joint_cam,
                'joint_valid': joint_valid,
                'smpl_param': smpl_param,
                'root_joint_depth': root_joint_depth,
                'cam_param': cam_param})

        cprint('Loaded Human36M {} {} samples'.format(self.data_split, str(len(datalist))), 'red')
        return datalist

    def get_smpl_coord(self, smpl_param, cam_param, do_flip, img_shape):
        pose, shape, trans = smpl_param['pose'], smpl_param['shape'], smpl_param['trans']
        smpl_pose = torch.FloatTensor(pose).view(-1, 3)
        smpl_shape = torch.FloatTensor(shape).view(1, -1)  # smpl parameters (pose: 72 dimension, shape: 10 dimension)
        R, t = np.array(cam_param['R'], dtype=np.float32).reshape(3, 3), np.array(cam_param['t'], dtype=np.float32).reshape(3)  # camera rotation and translation

        # merge root pose and camera rotation
        root_pose = smpl_pose[self.root_joint_idx, :].numpy()
        root_pose, _ = cv2.Rodrigues(root_pose)
        root_pose, _ = cv2.Rodrigues(np.dot(R, root_pose))
        smpl_pose[self.root_joint_idx] = torch.from_numpy(root_pose).view(3)

        # flip smpl pose parameter (axis-angle)
        if do_flip:
            for pair in self.flip_pairs:
                if pair[0] < len(smpl_pose) and pair[1] < len(smpl_pose):  # face keypoints are already included in self.flip_pairs. However, they are not included in smpl_pose.
                    smpl_pose[pair[0], :], smpl_pose[pair[1], :] = smpl_pose[pair[1], :].clone(), smpl_pose[pair[0], :].clone()
            smpl_pose[:, 1:3] *= -1  # multiply -1 to y and z axis of axis-angle
        smpl_pose = smpl_pose.view(1, -1)

        # get mesh and joint coordinates
        smpl_mesh_coord, smpl_joint_coord = self.smpl.layer['neutral'](smpl_pose, smpl_shape)

        # incorporate face keypoints
        smpl_mesh_coord = smpl_mesh_coord.numpy().astype(np.float32).reshape(-1, 3)
        smpl_joint_coord = smpl_joint_coord.numpy().astype(np.float32).reshape(-1, 3)
        smpl_face_kps_coord = smpl_mesh_coord[self.face_kps_vertex, :].reshape(-1, 3)
        smpl_joint_coord = np.concatenate((smpl_joint_coord, smpl_face_kps_coord))

        # compenstate rotation (translation from origin to root joint was not cancled)
        smpl_trans = np.array(trans, dtype=np.float32).reshape(3)  # translation vector from smpl coordinate to h36m world coordinate
        smpl_trans = np.dot(R, smpl_trans[:, None]).reshape(1, 3) + t.reshape(1, 3) / 1000
        root_joint_coord = smpl_joint_coord[self.root_joint_idx].reshape(1, 3)
        smpl_trans = smpl_trans - root_joint_coord + np.dot(R, root_joint_coord.transpose(1, 0)).transpose(1, 0)
        smpl_mesh_coord = smpl_mesh_coord + smpl_trans
        smpl_joint_coord = smpl_joint_coord + smpl_trans

        # flip translation
        if do_flip:  # avg of old and new root joint should be image center.
            focal, princpt = cam_param['focal'], cam_param['princpt']
            flip_trans_x = 2 * (((img_shape[1] - 1) / 2. - princpt[0]) / focal[0] * (smpl_joint_coord[self.root_joint_idx, 2] * 1000)) / 1000 - 2 * smpl_joint_coord[self.root_joint_idx][0]
            smpl_mesh_coord[:, 0] += flip_trans_x
            smpl_joint_coord[:, 0] += flip_trans_x

        # change to mean shape if beta is too far from it
        smpl_shape[(smpl_shape.abs() > 3).any(dim=1)] = 0.

        # meter -> milimeter
        smpl_mesh_coord *= 1000
        smpl_joint_coord *= 1000
        return smpl_mesh_coord, smpl_joint_coord, smpl_pose[0].numpy(), smpl_shape[0].numpy()

    def get_fitting_error(self, h36m_joint, smpl_mesh, do_flip):
        h36m_joint = h36m_joint - h36m_joint[self.h36m_root_joint_idx, None, :]  # root-relative
        if do_flip:
            h36m_joint[:, 0] = -h36m_joint[:, 0]
            for pair in self.h36m_flip_pairs:
                h36m_joint[pair[0], :], h36m_joint[pair[1], :] = h36m_joint[pair[1], :].copy(), h36m_joint[pair[0], :].copy()

        h36m_from_smpl = np.dot(self.h36m_joint_regressor, smpl_mesh)
        h36m_from_smpl = h36m_from_smpl - np.mean(h36m_from_smpl, 0)[None, :] + np.mean(h36m_joint, 0)[None, :]  # translation alignment

        error = np.sqrt(np.sum((h36m_joint - h36m_from_smpl) ** 2, 1)).mean()
        return error

    def __len__(self):
        return len(self.datalist)

    def check(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox, smpl_param, cam_param = data['img_path'], data['img_shape'], data['bbox'], data['smpl_param'], data['cam_param']
        if smpl_param is None:
            print(i, img_path)

    def check_fit_error(self, idx, noparam, error):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox, smpl_param, cam_param = data['img_path'], data['img_shape'], data['bbox'], data['smpl_param'], data['cam_param']
        if smpl_param is None:
            noparam += 1
            print('noparam', noparam)
        else:
            smpl_mesh_cam, smpl_joint_cam, smpl_pose, smpl_shape = self.get_smpl_coord(smpl_param, cam_param, False, img_shape)
            e = self.get_fitting_error(data['joint_cam'], smpl_mesh_cam, False)
            if e > self.fitting_thr:
                error += 1
                print('error', error)
        return noparam, error


    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox, smpl_param, cam_param = data['img_path'], data['img_shape'], data['bbox'], data['smpl_param'], data['cam_param']
        # img
        img = load_img(img_path)
        if smpl_param is not None:
            mask = load_img(img_path.replace('images', 'mask'))[:, :, 0]
        else:
            mask = np.zeros([img.shape[0], img.shape[1]])
        img, img2bb_trans, bb2img_trans, rot, do_flip, scale, mask = augmentation(img, bbox, self.data_split, exclude_flip=True, input_img_shape=(self.size, self.size), mask=mask)
        if self.color_aug is not None:
            img = self.color_aug(img)
        img = base_transform(img, self.size)

        # h36m gt
        h36m_joint_img = data['joint_img']
        h36m_joint_cam = data['joint_cam']
        h36m_joint_img_xy1 = np.concatenate((h36m_joint_img[:, :2], np.ones_like(h36m_joint_img[:, :1])), 1)
        h36m_joint_img[:, :2] = np.dot(img2bb_trans, h36m_joint_img_xy1.transpose(1, 0)).transpose(1, 0)

        # smpl coordinates
        smpl_mesh_cam, smpl_joint_cam, smpl_pose, smpl_shape = self.get_smpl_coord(smpl_param, cam_param, do_flip, img_shape)
        focal, princpt = cam_param['focal'], cam_param['princpt']
        smpl_joint_img = cam2pixel(smpl_joint_cam, focal, princpt)

        # affine transform x,y coordinates, root-relative depth
        smpl_joint_img_xy1 = np.concatenate((smpl_joint_img[:, :2], np.ones_like(smpl_joint_img[:, :1])), 1)
        smpl_joint_img[:, :2] = np.dot(img2bb_trans, smpl_joint_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]

        # if fitted mesh is too far from h36m gt, discard it
        is_valid_fit = np.array([[True]])
        error = self.get_fitting_error(data['joint_cam'], smpl_mesh_cam, do_flip)
        if error > self.fitting_thr:
            is_valid_fit = np.array([[False]])

        # 3D data rotation augmentation
        rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                                [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                                [0, 0, 1]], dtype=np.float32)
        smpl_joint_cam = np.dot(rot_aug_mat, smpl_joint_cam.transpose(1, 0)).transpose(1, 0) / 1000  # milimeter to meter
        smpl_joint_cam_root = smpl_joint_cam[self.root_joint_idx, None]
        smpl_joint_cam = (smpl_joint_cam - smpl_joint_cam_root) / self.std.numpy()  # root-relative
        smpl_mesh_cam = np.dot(rot_aug_mat, smpl_mesh_cam.transpose(1, 0)).transpose(1, 0) / 1000  # milimeter to meter
        smpl_mesh_cam = (smpl_mesh_cam - smpl_joint_cam_root) / self.std.numpy()  # root-relative
        h36m_joint_cam = np.dot(rot_aug_mat, h36m_joint_cam.transpose(1, 0)).transpose(1, 0) / 1000  # milimeter to meter
        h36m_joint_cam_root = h36m_joint_cam[self.h36m_root_joint_idx, None, :]
        h36m_joint_cam = (h36m_joint_cam - h36m_joint_cam_root) / self.std.numpy()  # root-relative
        # K
        focal, princpt = np.array(cam_param['focal']), np.array([cam_param['princpt'][0], cam_param['princpt'][1], 1]).reshape(3, 1)
        princpt = np.dot(img2bb_trans, princpt)[:, 0]
        focal *= scale
        K = np.array([focal[0], 0, princpt[0], 0, focal[1], princpt[1], 0, 0, 1]).reshape(3, 3)
        uv_point = h36m_joint_img[:, :2]
        uv_map = uv2map(uv_point.astype(np.int), img.shape[1:])
        uv_map = cv2.resize(uv_map.transpose(1, 2, 0), (img.shape[2] // 2, img.shape[1] // 2)).transpose(2, 0, 1)
        mask = cv2.resize(mask, (img.shape[2] // 2, img.shape[1] // 2))
        img, mask, K, smpl_joint_cam, uv_point, uv_map, smpl_mesh_cam, smpl_joint_cam_root, h36m_joint_cam, h36m_joint_cam_root, is_valid_fit \
            = [torch.from_numpy(x).float() for x in [img, mask, K, smpl_joint_cam, uv_point, uv_map, smpl_mesh_cam, smpl_joint_cam_root, h36m_joint_cam, h36m_joint_cam_root, is_valid_fit]]

        gt = [smpl_mesh_cam]
        for ds in self.down_sample_list[:-1]:
            gt.append(Pool(gt[-1].unsqueeze(0), ds)[0])

        return {
            'img': img,
            'mask_gt': mask,
            'uv_point': uv_point,
            'uv_gt': uv_map,
            'xyz_gt': h36m_joint_cam,
            'xyz_root': h36m_joint_cam_root,
            'mesh_gt': gt,
            'mesh_root': smpl_joint_cam_root,
            'K': K,
            'is_valid': is_valid_fit
        }

    def visualization(self, data):
        gs = gridspec.GridSpec(1, 5)
        v0 = (data['mesh_gt'][0] * self.std + data['xyz_root']).numpy()
        xyz = (data['xyz_gt'] * self.std + data['xyz_root']).numpy()
        K = data['K'].numpy()
        uv_map = cv2.resize(data['uv_gt'].sum(dim=0).clamp(max=1).numpy(), (self.size, self.size))
        mesh_uv = projectPoints(v0, K)
        xyz_uv = projectPoints(xyz, K)
        fig = plt.figure()
        ax = fig.add_subplot(gs[0, 0])
        img = inv_base_tranmsform(data['img'].numpy())
        plt.imshow(img)
        plt.triplot(mesh_uv[:, 0], mesh_uv[:, 1], self.faces[0].astype(np.int64), lw=0.2)
        plt.title('mesh2uv')
        ax.axis('off')
        ax = fig.add_subplot(gs[0, 1])
        plt.imshow(img)
        plot_hand(ax, data['uv_point'].numpy()[:, ::-1], draw_kp=False, linewidth='2')
        plt.title('uv')
        ax.axis('off')
        ax = fig.add_subplot(gs[0, 2])
        plt.imshow(img)
        plot_hand(ax, xyz_uv[:, ::-1], draw_kp=False, linewidth='2')
        plt.title('xyz2uv')
        ax.axis('off')
        ax = fig.add_subplot(gs[0, 3])
        mask = data['mask_gt'].numpy().astype(np.uint8) * 255
        mask = cv2.resize(mask, (self.size, self.size))[:, :, None]
        mask = np.concatenate([np.zeros_like(mask), mask, mask], 2)
        img_mask = cv2.addWeighted(img, 0.7, mask, 0.9, 0)
        plt.imshow(img_mask)
        plt.title('mask')
        ax.axis('off')
        ax = fig.add_subplot(gs[0, 4])
        img[:, :, 0] = (uv_map * 255).astype(np.uint8)
        plt.imshow(img)
        plt.title('uv_map')
        ax.axis('off')
        plt.savefig('read_sample.jpg')


if __name__ == '__main__':
    from options.base_options import BaseOptions
    import pickle
    import utils
    import os
    cur_dir = osp.dirname(osp.realpath(__file__))

    args = BaseOptions().parse()
    args.size = 256
    args.color_aug = True

    with open(os.path.join(cur_dir, '../../template/transform_body.pkl'), 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')
    down_transform_list = [
        utils.utils.to_sparse(down_transform)
        for down_transform in tmp['down_transform']
    ]

    dataset = Human36M(os.path.join(cur_dir,'../../data/Human36M'), 'train', args, down_transform_list, tmp['face'])
    for i in range(2000, 2001, 1):
        data = dataset.__getitem__(i)
        dataset.visualization(data)
