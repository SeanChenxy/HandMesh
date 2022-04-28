import os
import time
import torch
import json
from glob import glob
import logging
logging.getLogger('PIL').setLevel(logging.WARNING)


class Writer:
    def __init__(self, args=None):
        self.args = args
        if self.args is not None:
            log_filename = os.path.join(
                    args.out_dir, 'log.log')

            logging.basicConfig(
                    filename=log_filename,
                    level=logging.DEBUG,
                    format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')


    def print_str(self, info):
        logging.info(info)

    def print_info(self, info):
        message = 'Epoch: {}/{}, Duration: {:.3f}s, Train Loss: {:.4f}, Test Loss: {:.4f}' \
                .format(info['current_epoch'], info['epochs'], info['t_duration'], info['train_loss'], info['test_loss'])
        logging.info(message)

    def print_step(self, info):
        message = 'Epoch: {}/{}, Step: {}/{}, Total_step: {}, Duration: {:.3f}s, Train Loss: {:.4f}, L1 Loss: {:.4f}, Lr: {:.6f}' \
            .format(info['epoch'], info['max_epoch'], info['step'], info['max_step'], info['total_step'], info['step_duration'], info['train_loss'], info['l1_loss'], info['lr'])
        logging.info(message)

    def print_step_ft(self, info):
        message = 'Epoch: {}/{}, Step: {}/{}, Total: {}, Dur: {:.3f}s, FDur: {:.3f}s, BDur: {:.3f}s,, Train Loss: {:.4f}, L1 Loss: {:.4f}, Lr: {:.6f}' \
            .format(info['epoch'], info['max_epoch'], info['step'], info['max_step'], info['total_step'],
            info['step_duration'], info['forward_duration'] ,info['backward_duration'], info['train_loss'], info['l1_loss'], info['lr'])
        logging.info(message)

    def save_checkpoint(self, model, optimizer, scheduler, epoch, best=False, last=False):
        if best:
            save_path = os.path.join(self.args.checkpoints_dir, 'checkpoint_best.pt')
        elif last:
            save_path = os.path.join(self.args.checkpoints_dir, 'checkpoint_last.pt')
        else:
            save_path = os.path.join(self.args.checkpoints_dir, 'checkpoint_{:03d}.pt'.format(epoch))
        scheduler_state_dict = {} if scheduler is None else scheduler.state_dict()
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler_state_dict,
            }, save_path)
