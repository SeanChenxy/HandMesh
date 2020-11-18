import os
import torch
from utils.vis import cnt_area
import numpy as np
import cv2
from utils.vis import registration, map2uv, inv_base_transform, base_transform
from utils.draw3d import save_a_image_with_mesh_joints
from utils.read import save_mesh
import json
from datasets.FreiHAND.kinematics import mano_to_mpii
from utils.progress.bar import Bar
from termcolor import colored, cprint
import pickle


class Runner(object):
    def __init__(self, args, model, faces, device):
        super(Runner, self).__init__()
        self.args = args
        self.model = model
        self.eval_loader = None
        self.device = device
        self.faces = faces
        with open(os.path.join(self.args.work_dir, 'template', 'MANO_RIGHT.pkl'), 'rb') as f:
            mano = pickle.load(f, encoding='latin1')
        self.j_regressor = np.zeros([21, 778])
        self.j_regressor[:16] = mano['J_regressor'].toarray()
        for k, v in {16: 333, 17: 444, 18: 672, 19: 555, 20: 744}.items():
            self.j_regressor[k, v] = 1
        self.std = torch.tensor(0.2).to(device)
        self.face = torch.from_numpy(self.faces[0].astype(np.int64)).to(self.device)

    def set_eval_loader(self, eval_loader):
        self.eval_loader = eval_loader

    def run(self):
        if self.args.phase == 'eval':
            self.evaluation()
        elif self.args.phase == 'demo':
            self.demo()

    def phrase_data(self, data):
        for key, val in data.items():
            if isinstance(val, list):
                data[key] = [d.to(self.device) for d in data[key]]
            else:
                data[key] = data[key].to(self.device)
        return data

    def evaluation(self):
        if self.eval_loader is None:
            raise Exception('Please set_eval_loader before evaluation')
        args = self.args
        self.model.eval()
        xyz_pred_list, verts_pred_list = list(), list()
        bar = Bar(colored("EVAL", color='green'), max=len(self.eval_loader))
        with torch.no_grad():
            for step, data in enumerate(self.eval_loader):
                data = self.phrase_data(data)
                out = self.model(data['img'])
                # silhouette
                mask_pred = out.get('mask_pred')
                if mask_pred is not None:
                    mask_pred = (mask_pred[0] > 0.3).cpu().numpy().astype(np.uint8)
                    mask_pred = cv2.resize(mask_pred, (data['img'].size(3), data['img'].size(2)))
                    try:
                        contours, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        contours.sort(key=cnt_area, reverse=True)
                        poly = contours[0].transpose(1, 0, 2).astype(np.int32)
                    except:
                        poly = None
                else:
                    mask_pred = np.zeros([data['img'].size(3), data['img'].size(2)])
                    poly = None
                # vertex
                pred = out['mesh_pred'][0] if isinstance(out['mesh_pred'], list) else out['mesh_pred']
                vertex = (pred[0].cpu() * self.std.cpu()).numpy()
                uv_point_pred, uv_pred_conf = map2uv(out['uv_pred'].cpu().numpy(), (data['img'].size(2), data['img'].size(3)))
                vertex, align_state = registration(vertex, uv_point_pred[0], self.j_regressor, data['K'][0].cpu().numpy(), args.size, uv_conf=uv_pred_conf[0], poly=poly)

                vertex2xyz = mano_to_mpii(np.matmul(self.j_regressor, vertex))
                xyz_pred_list.append(vertex2xyz)
                verts_pred_list.append(vertex)
                # save_a_image_with_mesh_joints(inv_base_transform(data['img'][0].cpu().numpy())[:, :, ::-1], mask_pred, poly, data['K'][0].cpu().numpy(), vertex, self.faces[0], uv_point_pred[0], vertex2xyz,
                #                               os.path.join(args.out_dir, 'eval', str(step) + '_plot.jpg'))
                bar.suffix = '({batch}/{size})' .format(batch=step+1, size=len(self.eval_loader))
                bar.next()
        bar.finish()
        # save to a json
        xyz_pred_list = [x.tolist() for x in xyz_pred_list]
        verts_pred_list = [x.tolist() for x in verts_pred_list]
        with open(os.path.join(args.out_dir, args.exp_name + '.json'), 'w') as fo:
            json.dump([xyz_pred_list, verts_pred_list], fo)
        cprint('Save json file at ' + os.path.join(args.out_dir, args.exp_name + '.json'), 'green')

    def demo(self):
        args = self.args
        self.model.eval()
        image_fp = os.path.join(args.work_dir, 'images')
        image_files = [os.path.join(image_fp, i) for i in os.listdir(image_fp) if '_img.jpg' in i]
        bar = Bar(colored("DEMO", color='blue'), max=len(image_files))
        with torch.no_grad():
            for step, image_path in enumerate(image_files):
                image_name = image_path.split('/')[-1].split('_')[0]
                image = cv2.imread(image_path)[..., ::-1]
                input = torch.from_numpy(base_transform(image, size=224)).unsqueeze(0)
                K = np.load(image_path.replace('_img.jpg', '_K.npy'))

                out = self.model(input)
                # silhouette
                mask_pred = out.get('mask_pred')
                if mask_pred is not None:
                    mask_pred = (mask_pred[0] > 0.3).cpu().numpy().astype(np.uint8)
                    mask_pred = cv2.resize(mask_pred, (input.size(3), input.size(2)))
                    try:
                        contours, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        contours.sort(key=cnt_area, reverse=True)
                        poly = contours[0].transpose(1, 0, 2).astype(np.int32)
                    except:
                        poly = None
                else:
                    mask_pred = np.zeros([input.size(3), input.size(2)])
                    poly = None
                # vertex
                pred = out['mesh_pred'][0] if isinstance(out['mesh_pred'], list) else out['mesh_pred']
                vertex = (pred[0].cpu() * self.std.cpu()).numpy()
                uv_point_pred, uv_pred_conf = map2uv(out['uv_pred'].cpu().numpy(), (input.size(2), input.size(3)))
                vertex, align_state = registration(vertex, uv_point_pred[0], self.j_regressor, K, args.size, uv_conf=uv_pred_conf[0], poly=poly)

                vertex2xyz = mano_to_mpii(np.matmul(self.j_regressor, vertex))
                save_a_image_with_mesh_joints(image[..., ::-1], mask_pred, poly, K, vertex, self.faces[0], uv_point_pred[0], vertex2xyz,
                                              os.path.join(args.out_dir, 'demo', image_name + '_plot.jpg'))
                save_mesh(os.path.join(args.out_dir, 'demo', image_name + '_mesh.ply'), vertex, self.faces[0])

                bar.suffix = '({batch}/{size})' .format(batch=step+1, size=len(image_files))
                bar.next()
        bar.finish()
