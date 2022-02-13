import torch

from data import create_dataloader, create_dataset
import model as Model
from utils import util

import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from tensorboardX import SummaryWriter
import os
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/configuration_infer.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('Validation', opt['path']['log'], 'val', level=logging.INFO)
    
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = create_dataset(dataset_opt, phase)
            val_loader = create_dataloader(val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    logger.info('Begin Model Validation.')
    current_step = 0
    current_epoch = 0
    idx = 0

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)

    idx_d=0
    psnr_rlt = {}
    psnr_rlt_avg = {}
    psnr_total_avg = 0.

    ssim_rlt = {}
    ssim_rlt_avg = {}
    ssim_total_avg = 0.

    while idx_d < total_val_num:
        for _,  val_data in enumerate(val_loader):
            idx += 1

            idx_d+=1
            if(idx_d > total_val_num):
                break

            folder = val_data['key'][0]
            if psnr_rlt.get(folder, None) is None:
                psnr_rlt[folder] = []

            if ssim_rlt.get(folder, None) is None:
                ssim_rlt[folder] = []

            diffusion.feed_data(val_data)
            diffusion.test(continuous=False)
            visuals = diffusion.get_current_visuals()

            hr_img = util.tensor2img(visuals['HR'])  
            fake_img = util.tensor2img(visuals['INF']) 
            rlt_img = util.tensor2img(visuals['SR'])
            lr_img = util.tensor2img(visuals['LR'])

            psnr = util.calculate_psnr(fake_img, hr_img)
            psnr_rlt[folder].append(psnr)

            ssim = util.calculate_ssim(fake_img, hr_img)
            ssim_rlt[folder].append(ssim)

            Metrics.save_img(
                rlt_img, '{}/{}_{}_rlt.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                lr_img, '{}/{}_{}_lrs.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

    for k, v in psnr_rlt.items():
        psnr_rlt_avg[k] = sum(v) / len(v)
        psnr_total_avg += psnr_rlt_avg[k]
        psnr_total_avg /= len(psnr_rlt)

    for k, v in ssim_rlt.items():
        ssim_rlt_avg[k] = sum(v) / len(v)
        ssim_total_avg += ssim_rlt_avg[k]
        ssim_total_avg /= len(ssim_rlt)

    logger.info('# Validation # PSNR: {:.4e}'.format(psnr_total_avg))
    logger.info('# Validation # SSIM: {:.4e}'.format(ssim_total_avg))