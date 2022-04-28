import os
import numpy as np
import time
import torch
import cv2
import json
from utils.warmup_scheduler import adjust_learning_rate
from utils.vis import inv_base_tranmsform
from utils.zimeval import EvalUtil
from utils.transforms import rigid_align
from mobrecon.tools.vis import perspective, compute_iou, cnt_area
from mobrecon.tools.kinematics import mano_to_mpii, MPIIHandJoints
from mobrecon.tools.registration import registration
import vctoolkit as vc


class Runner(object):
    def __init__(self, cfg, args, model, train_loader, val_loader, test_loader, optimizer, writer, device, board, start_epoch=0):
        super(Runner, self).__init__()
        self.cfg = cfg
        self.args = args
        self.model = model
        face = np.load(os.path.join(cfg.MODEL.MANO_PATH, 'right_faces.npy'))
        self.face = torch.from_numpy(face).long()
        self.j_reg = np.load(os.path.join(self.cfg.MODEL.MANO_PATH, 'j_reg.npy'))
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.max_epochs = cfg.TRAIN.EPOCHS
        self.optimizer = optimizer
        self.writer = writer
        self.device = device
        self.board = board
        self.start_epoch = start_epoch
        self.epoch = max(start_epoch - 1, 0)
        if cfg.PHASE == 'train':
            self.total_step = self.start_epoch * (len(self.train_loader.dataset) // cfg.TRAIN.BATCH_SIZE)
            try:
                self.loss = self.model.loss
            except:
                self.loss = self.model.module.loss
        self.best_val_loss = np.float('inf')
        print('runner init done')

    def run(self):
        if self.cfg.PHASE == 'train':
            if self.val_loader is not None and self.epoch > 0:
                self.best_val_loss = self.eval()
            for epoch in range(self.start_epoch, self.max_epochs + 1):
                self.epoch = epoch
                t = time.time()
                if self.args.world_size > 1:
                    self.train_loader.sampler.set_epoch(epoch)
                train_loss = self.train()
                t_duration = time.time() - t
                if self.val_loader is not None:
                    val_loss = self.eval()
                else:
                    val_loss = np.float('inf')

                info = {
                    'current_epoch': self.epoch,
                    'epochs': self.max_epochs,
                    'train_loss': train_loss,
                    'test_loss': val_loss,
                    't_duration': t_duration
                }

                self.writer.print_info(info)
                if val_loss < self.best_val_loss:
                    self.writer.save_checkpoint(self.model, self.optimizer, None, self.epoch, best=True)
                    self.best_test_loss = val_loss
                self.writer.save_checkpoint(self.model, self.optimizer, None, self.epoch, last=True)
            self.pred()
        elif self.cfg.PHASE == 'eval':
            self.eval()
        elif self.cfg.PHASE == 'pred':
            self.pred()
        else:
            raise Exception('PHASE ERROR')

    def phrase_data(self, data):
        for key, val in data.items():
            try:
                if isinstance(val, list):
                    data[key] = [d.to(self.device) for d in data[key]]
                else:
                    data[key] = data[key].to(self.device)
            except:
                pass
        return data

    def board_scalar(self, phase, n_iter, lr=None, **kwargs):
        split = '/'
        for key, val in kwargs.items():
            if 'loss' in key:
                if isinstance(val, torch.Tensor):
                    val = val.item()
                self.board.add_scalar(phase + split + key, val, n_iter)
            if lr:
                self.board.add_scalar(phase + split + 'lr', lr, n_iter)

    def draw_results(self, data, out, loss, batch_id, aligned_verts=None):
        img_cv2 = inv_base_tranmsform(data['img'][batch_id].cpu().numpy())[..., :3]
        draw_list = []
        if 'joint_img' in data:
            draw_list.append( vc.render_bones_from_uv(np.flip(data['joint_img'][batch_id, :, :2].cpu().numpy()*self.cfg.DATA.SIZE, axis=-1).copy(),
                                                      img_cv2.copy(), MPIIHandJoints, thickness=2) )
        if 'joint_img' in out:
            try:
                draw_list.append( vc.render_bones_from_uv(np.flip(out['joint_img'][batch_id, :, :2].detach().cpu().numpy()*self.cfg.DATA.SIZE, axis=-1).copy(),
                                                         img_cv2.copy(), MPIIHandJoints, thickness=2) )
            except:
                draw_list.append(img_cv2.copy())
        if 'root' in data:
            root = data['root'][batch_id:batch_id+1, :3]
        else:
            root = torch.FloatTensor([[0, 0, 0.6]]).to(data['img'].device)
        if 'verts' in data:
            vis_verts_gt = img_cv2.copy()
            verts = data['verts'][batch_id:batch_id+1, :, :3] * 0.2 + root
            vp = perspective(verts.permute(0, 2, 1), data['calib'][batch_id:batch_id+1, :4])[0].cpu().numpy().T
            for i in range(vp.shape[0]):
                cv2.circle(vis_verts_gt, (int(vp[i, 0]), int(vp[i, 1])), 1, (255, 0, 0), -1)
            draw_list.append(vis_verts_gt)
        if 'verts' in out:
            try:
                vis_verts_pred = img_cv2.copy()
                if aligned_verts is None:
                    verts = out['verts'][batch_id:batch_id+1, :, :3] * 0.2 + root
                else:
                    verts = aligned_verts
                vp = perspective(verts.permute(0, 2, 1), data['calib'][batch_id:batch_id+1, :4])[0].detach().cpu().numpy().T
                for i in range(vp.shape[0]):
                    cv2.circle(vis_verts_pred, (int(vp[i, 0]), int(vp[i, 1])), 1, (255, 0, 0), -1)
                draw_list.append(vis_verts_pred)
            except:
                draw_list.append(img_cv2.copy())

        return np.concatenate(draw_list, 1)

    def board_img(self, phase, n_iter, data, out, loss, batch_id=0):
        draw = self.draw_results(data, out, loss, batch_id)
        self.board.add_image(phase + '/res', draw.transpose(2, 0, 1), n_iter)

    def train(self):
        self.writer.print_str('TRAINING ..., Epoch {}/{}'.format(self.epoch, self.max_epochs))
        self.model.train()
        total_loss = 0
        forward_time = 0.
        backward_time = 0.
        start_time = time.time()
        for step, data in enumerate(self.train_loader):
            ts = time.time()
            adjust_learning_rate(self.optimizer, self.epoch, step, len(self.train_loader), self.cfg.TRAIN.LR, self.cfg.TRAIN.LR_DECAY, self.cfg.TRAIN.DECAY_STEP, self.cfg.TRAIN.WARMUP_EPOCHS)
            data = self.phrase_data(data)
            self.optimizer.zero_grad()
            out = self.model(data['img'])
            tf = time.time()
            forward_time += tf - ts
            losses = self.loss(verts_pred=out.get('verts'),
                               joint_img_pred=out['joint_img'],
                               verts_gt=data.get('verts'),
                               joint_img_gt=data['joint_img'],
                               face=self.face,
                               aug_param=(None, data.get('aug_param'))[self.epoch>4],
                               bb2img_trans=data.get('bb2img_trans'),
                               size=data['img'].size(2),
                               mask_gt=data.get('mask'),
                               trans_pred=out.get('trans'),
                               alpha_pred=out.get('alpha'),
                               img=data.get('img'))
            loss = losses['loss']
            loss.backward()
            self.optimizer.step()
            tb = time.time()
            backward_time +=  tb - tf

            self.total_step += 1
            total_loss += loss.item()
            if self.board is not None:
                self.board_scalar('train', self.total_step, self.optimizer.param_groups[0]['lr'], **losses)
            if self.total_step % 100 == 0:
                cur_time = time.time()
                duration = cur_time - start_time
                start_time = cur_time
                info = {
                    'train_loss': loss.item(),
                    'l1_loss': losses.get('verts_loss', 0),
                    'epoch': self.epoch,
                    'max_epoch': self.max_epochs,
                    'step': step,
                    'max_step': len(self.train_loader),
                    'total_step': self.total_step,
                    'step_duration': duration,
                    'forward_duration': forward_time,
                    'backward_duration': backward_time,
                    'lr': self.optimizer.param_groups[0]['lr']
                }
                self.writer.print_step_ft(info)
                forward_time = 0.
                backward_time = 0.

        if self.board is not None:
            self.board_img('train', self.epoch, data, out, losses)

        return total_loss / len(self.train_loader)

    def eval(self):
        self.writer.print_str('EVALING ... Epoch {}/{}'.format(self.epoch, self.max_epochs))
        self.model.eval()
        evaluator_2d = EvalUtil()
        evaluator_rel = EvalUtil()
        evaluator_pa = EvalUtil()
        mask_iou = []
        joint_cam_errors = []
        pa_joint_cam_errors = []
        joint_img_errors = []
        with torch.no_grad():
            for step, data in enumerate(self.val_loader):
                if self.board is None and step % 100 == 0:
                    print(step, len(self.val_loader))
                # get data then infernce
                data = self.phrase_data(data)
                out = self.model(data['img'])

                # get vertex pred
                verts_pred = out['verts'][0].cpu().numpy() * 0.2
                joint_cam_pred = mano_to_mpii(np.matmul(self.j_reg, verts_pred)) * 1000.0

                # get mask pred
                mask_pred = out.get('mask')
                if mask_pred is not None:
                    mask_pred = (mask_pred[0] > 0.3).cpu().numpy().astype(np.uint8)
                    mask_pred = cv2.resize(mask_pred, (data['img'].size(3), data['img'].size(2)))
                else:
                    mask_pred = np.zeros((data['img'].size(3), data['img'].size(2)), np.uint8)

                # get uv pred
                joint_img_pred = out.get('joint_img')
                if joint_img_pred is not None:
                    joint_img_pred = joint_img_pred[0].cpu().numpy() * data['img'].size(2)
                else:
                    joint_img_pred = np.zeros((21, 2), dtype=np.float)

                # pck
                joint_cam_gt = data['joint_cam'][0].cpu().numpy() * 1000.0
                joint_cam_align = rigid_align(joint_cam_pred, joint_cam_gt)
                evaluator_2d.feed(data['joint_img'][0].cpu().numpy() * data['img'].size(2), joint_img_pred)
                evaluator_rel.feed(joint_cam_gt, joint_cam_pred)
                evaluator_pa.feed(joint_cam_gt, joint_cam_align)

                # error
                if 'mask_gt' in data.keys():
                    mask_iou.append(compute_iou(mask_pred, cv2.resize(data['mask_gt'][0].cpu().numpy(), (data['img'].size(3), data['img'].size(2)))))
                else:
                    mask_iou.append(0)
                joint_cam_errors.append(np.sqrt(np.sum((joint_cam_pred - joint_cam_gt) ** 2, axis=1)))
                pa_joint_cam_errors.append(np.sqrt(np.sum((joint_cam_gt - joint_cam_align) ** 2, axis=1)))
                joint_img_errors.append(np.sqrt(np.sum((data['joint_img'][0].cpu().numpy()*data['img'].size(2) - joint_img_pred) ** 2, axis=1)))

            # get auc
            _1, _2, _3, auc_rel, pck_curve_rel, thresholds2050 = evaluator_rel.get_measures(20, 50, 20)
            _1, _2, _3, auc_pa, pck_curve_pa, _ = evaluator_pa.get_measures(20, 50, 20)
            _1, _2, _3, auc_2d, pck_curve_2d, _ = evaluator_2d.get_measures(0, 30, 20)
            # get error
            miou = np.array(mask_iou).mean()
            mpjpe = np.array(joint_cam_errors).mean()
            pampjpe = np.array(pa_joint_cam_errors).mean()
            uve = np.array(joint_img_errors).mean()

            if self.board is not None:
                self.board_scalar('test', self.epoch, **{'auc_loss': auc_rel, 'pa_auc_loss': auc_pa, '2d_auc_loss': auc_2d, 'mIoU_loss': miou, 'uve': uve, 'mpjpe_loss': mpjpe, 'pampjpe_loss': pampjpe})
                self.board_img('test', self.epoch, data, out, {})
            elif self.args.world_size < 2:
                print( f'pampjpe: {pampjpe}, mpjpe: {mpjpe}, uve: {uve}, miou: {miou}, auc_rel: {auc_rel}, auc_pa: {auc_pa}, auc_2d: {auc_2d}')
                print('thresholds2050', thresholds2050)
                print('pck_curve_all_pa', pck_curve_pa)
            self.writer.print_str( f'pampjpe: {pampjpe}, mpjpe: {mpjpe}, uve: {uve}, miou: {miou}, auc_rel: {auc_rel}, auc_pa: {auc_pa}, auc_2d: {auc_2d}')

        return pampjpe

    def pred(self):
        self.writer.print_str('PREDICING ... Epoch {}/{}'.format(self.epoch, self.max_epochs))
        self.model.eval()
        xyz_pred_list, verts_pred_list = list(), list()
        with torch.no_grad():
            for step, data in enumerate(self.test_loader):
                if self.board is None and step % 100 == 0:
                    print(step, len(self.test_loader))
                data = self.phrase_data(data)
                out = self.model(data['img'])
                # get verts pred
                verts_pred = out['verts'][0].cpu().numpy() * 0.2

                # get mask pred
                mask_pred = out.get('mask')
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
                    poly = None

                # get uv pred
                joint_img_pred = out.get('joint_img')
                if joint_img_pred is not None:
                    joint_img_pred = joint_img_pred[0].cpu().numpy() * data['img'].size(2)
                    verts_pred, align_state = registration(verts_pred, joint_img_pred, self.j_reg, data['calib'][0].cpu().numpy(), self.cfg.DATA.SIZE, poly=poly)

                # get joint_cam
                joint_cam_pred = mano_to_mpii(np.matmul(self.j_reg, verts_pred))

                # track data
                xyz_pred_list.append(joint_cam_pred)
                verts_pred_list.append(verts_pred)
                if self.cfg.TEST.SAVE_PRED:
                    draw = self.draw_results(data, out, {}, 0, aligned_verts=torch.from_numpy(verts_pred).float()[None, ...])[..., ::-1]
                    cv2.imwrite(os.path.join(self.args.out_dir, self.cfg.TEST.SAVE_DIR, f'{step}.png'), draw)

        # dump results
        xyz_pred_list = [x.tolist() for x in xyz_pred_list]
        verts_pred_list = [x.tolist() for x in verts_pred_list]
        # save to a json
        with open(os.path.join(self.args.out_dir, f'{self.args.exp_name}.json'), 'w') as fo:
            json.dump(
                [
                    xyz_pred_list,
                    verts_pred_list
                ], fo)
        self.writer.print_str('Dumped %d joints and %d verts predictions to %s' % (
            len(xyz_pred_list), len(verts_pred_list), os.path.join(self.args.work_dir, 'out', self.args.exp_name, f'{self.args.exp_name}.json')))
