import os
import torch
from utils.vis import cnt_area
import numpy as np
import cv2
from utils.vis import registration, map2uv, inv_base_tranmsform, base_transform, tensor2array
from utils.draw3d import save_a_image_with_mesh_joints
from utils.read import save_mesh
import json
from cmr.datasets.FreiHAND.kinematics import mano_to_mpii
from utils.progress.bar import Bar
from termcolor import colored, cprint
import pickle
import time
from utils.transforms import rigid_align


class Runner(object):
    def __init__(self, args, model, faces, device):
        super(Runner, self).__init__()
        self.args = args
        self.model = model
        self.faces = faces
        self.device = device
        self.face = torch.from_numpy(self.faces[0].astype(np.int64)).to(self.device)

    def set_train_loader(self, train_loader, epochs, optimizer, scheduler, writer, board, start_epoch=0):
        self.train_loader = train_loader
        self.max_epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = writer
        self.board = board
        self.start_epoch = start_epoch
        self.epoch = max(start_epoch - 1, 0)
        self.total_step = self.start_epoch * (len(self.train_loader.dataset) // self.writer.args.batch_size)
        self.loss = self.model.loss
        if self.args.dataset=='Human36M':
            self.j_regressor = self.train_loader.dataset.h36m_joint_regressor
            self.j_eval = self.train_loader.dataset.h36m_eval_joint
        else:
            self.j_regressor = self.train_loader.dataset.j_regressor
        self.std = train_loader.dataset.std.to(self.device)

    def set_eval_loader(self, eval_loader):
        self.eval_loader = eval_loader
        if self.args.phase != 'train':
            if self.args.dataset=='Human36M':
                self.j_regressor = self.eval_loader.dataset.h36m_joint_regressor
                self.j_eval = self.eval_loader.dataset.h36m_eval_joint
            else:
                self.j_regressor = self.eval_loader.dataset.j_regressor
            self.std = eval_loader.dataset.std.to(self.device)
            self.board = None
    
    def set_demo(self, args):
        with open(os.path.join(args.work_dir, '../template/MANO_RIGHT.pkl'), 'rb') as f:
            mano = pickle.load(f, encoding='latin1')
        self.j_regressor = np.zeros([21, 778])
        self.j_regressor[:16] = mano['J_regressor'].toarray()
        for k, v in {16: 333, 17: 444, 18: 672, 19: 555, 20: 744}.items():
            self.j_regressor[k, v] = 1
        self.std = torch.tensor(0.20)

    def train(self):
        best_error = np.float('inf')
        for epoch in range(self.start_epoch, self.max_epochs + 1):
            self.epoch = epoch
            t = time.time()
            train_loss = self.train_a_epoch()
            t_duration = time.time() - t
            self.scheduler.step()
            info = {
                    'current_epoch': self.epoch,
                    'epochs': self.max_epochs,
                    'train_loss': train_loss,
                    'test_loss': 0.0,
                    't_duration': t_duration
                }
            self.writer.print_info(info)
            if self.args.dataset=='Human36M':
                test_error = self.evaluation_withgt()
                if test_error < best_error:
                    best_error = test_error
                    self.writer.save_checkpoint(self.model, self.optimizer, self.scheduler, self.epoch, best=True)
            self.writer.save_checkpoint(self.model, self.optimizer, self.scheduler, self.epoch, last=True)
        if self.args.dataset=='FreiHAND' and self.eval_loader is not None:
            self.evaluation()

    def board_img(self, phase, n_iter, img, **kwargs):
        # print(rendered_mask.shape, rendered_mask.max(), rendered_mask.min())
        self.board.add_image(phase + '/img', tensor2array(img), n_iter)
        if kwargs.get('mask_pred') is not None:
            self.board.add_image(phase + '/mask_gt', tensor2array(kwargs['mask_gt'][0]), n_iter)
            self.board.add_image(phase + '/mask_pred', tensor2array(kwargs['mask_pred'][0]), n_iter)
        if kwargs.get('uv_pred') is not None:
            self.board.add_image(phase + '/uv_gt', tensor2array(kwargs['uv_gt'][0].sum(dim=0).clamp(max=1)), n_iter)
            self.board.add_image(phase + '/uv_pred', tensor2array(kwargs['uv_pred'][0].sum(dim=0).clamp(max=1)), n_iter)
        if kwargs.get('uv_prior') is not None:
            self.board.add_image(phase + '/uv_prior', tensor2array(kwargs['uv_prior'][0].sum(dim=0).clamp(max=1)), n_iter)

    def board_scalar(self, phase, n_iter, lr=None, **kwargs):
        for key, val in kwargs.items():
            if 'loss' in key:
                self.board.add_scalar(phase + '/' + key, val.item(), n_iter)
        if lr:
            self.board.add_scalar('lr', lr, n_iter)

    def phrase_data(self, data):
        for key, val in data.items():
            if isinstance(val, list):
                data[key] = [d.to(self.device) for d in data[key]]
            else:
                data[key] = data[key].to(self.device)
        return data

    def train_a_epoch(self):
        self.model.train()
        total_loss = 0
        bar = Bar(colored("TRAIN", color='blue'), max=len(self.train_loader))
        for step, data in enumerate(self.train_loader):
            t = time.time()
            data = self.phrase_data(data)
            self.optimizer.zero_grad()
            out = self.model(data['img'])
            loss = self.loss(pred=out['mesh_pred'], gt=data.get('mesh_gt'), uv_pred=out.get('uv_pred'), uv_gt=data.get('uv_gt'),
                             mask_pred=out.get('mask_pred'), mask_gt=data.get('mask_gt'), face=self.face,
                             uv_prior=out.get('uv_prior'), uv_prior2=out.get('uv_prior2'), mask_prior=out.get('mask_prior'))
            loss['loss'].backward()
            total_loss += loss['loss'].item()
            self.optimizer.step()
            step_duration = time.time() - t
            self.total_step += 1
            self.board_scalar('train', self.total_step, self.optimizer.param_groups[0]['lr'], **loss)
            bar.suffix = (
                '({epoch}/{max_epoch}:{batch}/{size}) '
                'time: {time:.3f} | '
                'loss: {loss:.4f} | '
                'l1_loss: {l1_loss:.4f} | '
                'lr: {lr:.6f} | '
            ).format(epoch=self.epoch, max_epoch=self.max_epochs, batch=step, size=len(self.train_loader),
                     loss=loss['loss'], l1_loss=loss['l1_loss'], time=step_duration,
                     lr=self.optimizer.param_groups[0]['lr'])
            bar.next()
            if self.total_step % 100 == 0:
                info = {
                    'train_loss': loss['loss'],
                    'l1_loss': loss.get('l1_loss', 0),
                    'epoch': self.epoch,
                    'max_epoch': self.max_epochs,
                    'step': step,
                    'max_step': len(self.train_loader),
                    'total_step': self.total_step,
                    'step_duration': step_duration,
                    'lr': self.optimizer.param_groups[0]['lr']
                }
                self.writer.print_step(info)

        bar.finish()
        self.board_img('train', self.epoch, data['img'][0], mask_gt=data.get('mask_gt'), mask_pred=out.get('mask_pred'), uv_gt=data.get('uv_gt'), uv_pred=out.get('uv_pred'), uv_prior=out.get('uv_prior'))
        return total_loss / len(self.train_loader)

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
                uv_pred = out['uv_pred']
                if uv_pred.ndim == 4:
                    uv_point_pred, uv_pred_conf = map2uv(uv_pred.cpu().numpy(), (data['img'].size(2), data['img'].size(3)))
                else:
                    uv_point_pred, uv_pred_conf = (uv_pred * args.size).cpu().numpy(), [None,]
                vertex, align_state = registration(vertex, uv_point_pred[0], self.j_regressor, data['K'][0].cpu().numpy(), args.size, uv_conf=uv_pred_conf[0], poly=poly)

                vertex2xyz = mano_to_mpii(np.matmul(self.j_regressor, vertex))
                xyz_pred_list.append(vertex2xyz)
                verts_pred_list.append(vertex)
                if args.phase == 'eval':
                    save_a_image_with_mesh_joints(inv_base_tranmsform(data['img'][0].cpu().numpy())[:, :, ::-1], mask_pred, poly, data['K'][0].cpu().numpy(), vertex, self.faces[0], uv_point_pred[0], vertex2xyz,
                                              os. path.join(args.out_dir, 'eval', str(step) + '_plot.jpg'))
                bar.suffix = '({batch}/{size})' .format(batch=step+1, size=len(self.eval_loader))
                bar.next()
        bar.finish()
        # save to a json
        xyz_pred_list = [x.tolist() for x in xyz_pred_list]
        verts_pred_list = [x.tolist() for x in verts_pred_list]
        with open(os.path.join(args.out_dir, args.exp_name + '.json'), 'w') as fo:
            json.dump([xyz_pred_list, verts_pred_list], fo)
        cprint('Save json file at ' + os.path.join(args.out_dir, args.exp_name + '.json'), 'green')

    def evaluation_withgt(self):
        # self.writer.print_str('Eval error on set')
        self.model.eval()
        joint_errors = []
        pa_joint_errors = []
        duration = [0,]
        bar = Bar(colored("TEST", color='yellow'), max=len(self.eval_loader))
        with torch.no_grad():
            for i, data in enumerate(self.eval_loader):
                data = self.phrase_data(data)
                t1 = time.time()
                out = self.model(data['img'])
                torch.cuda.synchronize()
                if i > 10:
                    duration.append((time.time()-t1)*1000)
                gt = data['mesh_gt'][0] if isinstance(data['mesh_gt'], list) else data['mesh_gt']
                xyz_gt = data['xyz_gt']
                pred = out['mesh_pred'][0] if isinstance(out['mesh_pred'], list) else out['mesh_pred']
                pred = (pred[0].cpu() * self.std.cpu()).numpy()
                joint_pred = np.dot(self.j_regressor, pred)
                gt = (gt[0].cpu() * self.std.cpu()).numpy()
                xyz_gt = (xyz_gt[0].cpu() * self.std.cpu()).numpy()

                rel_joint_pred = joint_pred[self.j_eval, :] * 1000
                rel_joint_gt = xyz_gt[self.j_eval, :] * 1000
                joint_errors.append(np.sqrt(np.sum((rel_joint_gt - rel_joint_pred) ** 2, axis=1)))
                pa_joint_errors.append(np.sqrt(np.sum((rel_joint_gt - rigid_align(rel_joint_pred, rel_joint_gt)) ** 2, axis=1)))

                bar.suffix = (
                    '({batch}/{size}) '
                    'MPJPE:{j:.3f} '
                    'PA-MPJPE:{pa_j:.3f} '
                    'T:{t:.0f}'
                ).format(batch=i, size=len(self.eval_loader), j=np.array(joint_errors).mean(), pa_j=np.array(pa_joint_errors).mean(), t=np.array(duration).mean())
                bar.next()
        bar.finish()

        j_error = np.array(joint_errors).mean()
        pa_j_error = np.array(pa_joint_errors).mean()
        if self.board is not None:
            self.board_scalar('test', self.epoch, **{'j_loss': j_error, 'pa_j_loss': pa_j_error})
            self.board_img('test', self.epoch, data['img'][0], uv_gt=data['uv_gt'], uv_pred=out['uv_pred'], mask_gt=data.get('mask_gt'), mask_pred=out.get('mask_pred'))

        return pa_j_error

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
                image = cv2.resize(image, (args.size, args.size))
                input = torch.from_numpy(base_transform(image, size=args.size)).unsqueeze(0).to(self.device)
                K = np.load(image_path.replace('_img.jpg', '_K.npy'))
                K[0, 0] = K[0, 0] / 224 * args.size
                K[1, 1] = K[1, 1] / 224 * args.size
                K[0, 2] = args.size // 2
                K[1, 2] = args.size // 2

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
                uv_pred = out['uv_pred']
                if uv_pred.ndim == 4:
                    uv_point_pred, uv_pred_conf = map2uv(uv_pred.cpu().numpy(), (input.size(2), input.size(3)))
                else:
                    uv_point_pred, uv_pred_conf = (uv_pred * args.size).cpu().numpy(), [None,]
                vertex, align_state = registration(vertex, uv_point_pred[0], self.j_regressor, K, args.size, uv_conf=uv_pred_conf[0], poly=poly)

                vertex2xyz = mano_to_mpii(np.matmul(self.j_regressor, vertex))
                save_a_image_with_mesh_joints(image[..., ::-1], mask_pred, poly, K, vertex, self.faces[0], uv_point_pred[0], vertex2xyz,
                                              os.path.join(args.out_dir, 'demo', image_name + '_plot.jpg'))
                save_mesh(os.path.join(args.out_dir, 'demo', image_name + '_mesh.ply'), vertex, self.faces[0])

                bar.suffix = '({batch}/{size})' .format(batch=step+1, size=len(image_files))
                bar.next()
        bar.finish()
