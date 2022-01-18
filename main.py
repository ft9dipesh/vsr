import torch

import argparse
import core.metrics as Metrics
import model as Model

import os
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='config/config.json', help='JSON file for configuration')
parser.add_argument('p', '--phase', type=str, choices=['train','val'],
                    help='Run either train(training) or val(generation)',
                    default='train')
parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)


diffusion = Model.create_model(opt)

current_step = diffusion.begin_step
current_epoch = diffusion.begin_epoch
n_iter = opt['train']['n_iter']

diffusion.set_new_noise_schedule(
    opt['model']['beta_schedule'][opt['phase']],
    schedule_phase=opt['phase']
)

if opt['phase'] == 'train':
    while current_step < n_iter:
        current_epoch += 1
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > n_iter:
                break
            diffusion.feed_data(train_data)
            diffusion.ooptimize_parameters()

            if current_step % opt['train']['val_freq'] == 0:
                avg_psnr = 0.0
                idx = 0
                result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                os.makedirs(result_path, exist_ok=True)

                diffusion.set_new_noise_schedule(
                    opt['model']['beta_schedule']['val'],
                    schedule_phase='val',
                )
                for _, val_data in enumerate(val_loader):
                    idx += 1
                    diffusion.feed_data(val_data)
                    diffusion.test(continuous=False)
                    visuals = diffusion.get_current_visuals()
                    sr_img = Metrics.tensor2img(visuals['SR'])
                    hr_img = Metrics.tensor2img(visuals['HR'])
                    lr_img = Metrics.tensor2img(visuals['LR'])
                    fake_img = Metrics.tensor2img(visuals['INF'])

                    avg_psnr += Metrics.calculate_psnr(sr_img, hr_img)

                avg_psnr = avg_psnr / idx
                diffusion.set_new_noise_schedule(
                    opt['model']['beta_schedule']['train'],
                    schedule_phase='train'
                )

else:
    avg_psnr = 0.0
    avg_ssim = 0.0
    idx = 0
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    for _, val_data in enumerate(val_loader):
        idx+=1
        diffusion.feed_data(val_data)
        diffusion.test(continuous=False)
        visuals = diffusion.get_current_visuals()

        hr_img = Metrics.tensor2img(visuals['HR'])
        lr_img = Metrics.tensor2img(visuals['LR'])
        fake_img = Metrics.tensor2img(visuals['INF'])

        sr_img_mode = 'grid'
        if sr_img_mode == 'single':
            # single img series
            sr_img = visuals['SR']  # uint8
            sample_num = sr_img.shape[0]
            for iter in range(0, sample_num):
                Metrics.save_img(
                    Metrics.tensor2img(sr_img[iter]),
                    '{}/{}_{}_sr_{}.png'.format(
                        result_path,
                        current_step,
                        idx,
                        iter,
                    )
                )
        else:
            # grid img
            sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
            Metrics.save_img(
                sr_img,
                '{}/{}_{}_sr_process.png'.format(
                    result_path,
                    current_step,
                    idx,
                )
            )
            Metrics.save_img(
                Metrics.tensor2img(visuals['SR'][-1]),
                '{}/{}_{}_sr.png'.format(
                    result_path,
                    current_step,
                    idx,
                )
            )

        Metrics.save_img(
            hr_img,
            '{}/{}_{}_hr.png'.format(result_path, current_step, idx),
        )
        Metrics.save_img(
            lr_img,
            '{}/{}_{}_lr.png'.format(result_path, current_step, idx),
        )
        Metrics.save_img(
            fake_img,
            '{}/{}_{}_inf.png'.format(result_path, current_step, idx),
        )

        eval_psnr = Matrics.calculate_psnr(
            Metrics.tensor2img(visuals['SR'][-1]),
            hr_img,
        )
        eval_ssim = Metrics.calculate_ssim(
            Metrics.tensor2img(visuals['SR'][-1]),
            hr_img,
        )

        avg_psnr += eval_psnr
        avg_ssim += eval_ssim

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx
