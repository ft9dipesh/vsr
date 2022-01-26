from multiprocessing.dummy import current_process
import torch

from data import create_dataloader, create_dataset
from utils import util
import model as Model

import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from tensorboardX import SummaryWriter
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/configuration.json', 
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either training or generation', default='train')
    parser.add_argument('-g', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')

    args = parser.parse_args()
    opt = Logger.parse(args)

    opt = Logger.dict_to_nonedict(opt)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger("Validation", opt['path']['log'], 'val', level=logging.INFO)

    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # DATASET
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt, phase)
            train_loader = create_dataloader(train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = create_dataset(dataset_opt, phase)
            val_loader = create_dataloader(val_set, dataset_opt, phase)
    
    logger.info('Initial Dataset finished')

    # MODEL
    diffusion = Model.create_model(opt)
    logger.info('Initial Model finished')

    """ TRAINING """

    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}'.format(
            current_epoch,
            current_step
        ))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']],
        schedule_phase=opt['phase']
    )

    if opt['phase'] == 'train':
        while current_step < n_iter:
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()

                # LOG
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(current_epoch, current_step)
                    for k,v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                if current_step % opt['train']['val_freq'] == 0:
                    pbar = util.ProgressBar(len(val_loader))
                    psnr_rlt = {}
                    psnr_rlt_avg = {}
                    psnr_total_avg = 0.
                    
                    result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    for val_data in val_loader:
                        folder = val_data['folder'][0]
                        idx_d = val_data['idx'].item()
                        if psnr_rlt.get(folder, None) is None:
                            psnr_rlt[folder] = []

                        diffusion.feed_data(val_data)
                        diffusion.test(continuous=False)
                        visuals = diffusion.get_current_visuals()
                        rlt_img = util.tensor2img(visuals['SR'])
                        gt_img = util.tensor2img(visuals['HR'])
                        lr_img = util.tensor2img(visuals['LR'])
                        fake_img = util.tensor2img(visuals['INF'])
                        
                        Metrics.save_img(
                            gt_img, '{}/{}_{}_gt.png'.format(result_path, current_step, idx_d)
                        )
                        Metrics.save_img(
                            rlt_img, '{}/{}_{}_rlt.png'.format(result_path, current_step, idx_d)
                        )
                        Metrics.save_img(
                            lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx_d)
                        )
                        Metrics.save_img(
                            fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx_d)
                        )
                        tb_logger.add_image(
                            'Iter_{}'.format(current_step),
                            np.transpose(
                                np.concatenate(
                                    (fake_img, rlt_img, gt_img),
                                    axis=1
                                ),
                                [2,0,1]
                            ),
                            idx_d
                        )

                        psnr = util.calculate_psnr(rlt_img, gt_img)
                        psnr_rlt[folder].append(psnr)
                        pbar.update('Test {} - {}'.format(folder, idx_d))

                    for k, v in psnr_rlt.items():
                        psnr_rlt_avg[k] = sum(v) / len(v)
                        psnr_total_avg += psnr_rlt_avg[k]
                    psnr_total_avg /= len(psnr_rlt)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'],
                        schedule_phase='train'
                    )

                    logger.info('# Validation # PSNR: {:.4e}'.format(psnr_total_avg))
                    logger_val = logging.getLogger('val')
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(current_epoch, current_step, psnr_total_avg))
                    tb_logger.add_scalar('psnr', psnr_total_avg, current_step)

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states')
                    diffusion.save_network(current_epoch, current_step)
        
        logger.info('End of training')

if __name__ == "__main__":
    main()